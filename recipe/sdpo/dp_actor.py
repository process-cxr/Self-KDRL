# Copyright 2025 SDPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SDPO Data Parallel Actor.

This module provides the SDPO-specific actor implementation that extends
verl's DataParallelPPOActor with self-distillation capabilities.
"""

import logging
import os
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.workers.actor import DataParallelPPOActor
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs

# Use extended get_policy_loss_fn that supports 'sdpo' mode
from .core_algos import get_policy_loss_fn, compute_self_distillation_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    """
    Trust-region teacher regularization module.

    This module creates a teacher by linearly interpolating between
    a reference model and a student model, limiting how far the teacher
    can deviate from the reference.

    Args:
        ref_module: Reference model (frozen)
        student_module: Student model (trainable)
        mix_coef: Mixing coefficient for interpolation
    """

    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        """Forward pass with interpolated logits."""
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)

        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]

        # Linear interpolation: logits = (1 - mix_coef) * ref + mix_coef * student
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class SDPODataParallelPPOActor(DataParallelPPOActor):
    """
    SDPO-specific Data Parallel PPO Actor.

    This class extends verl's DataParallelPPOActor with self-distillation
    capabilities for SDPO training.

    Key additions:
    - Teacher module management (EMA or trust-region)
    - Self-distillation loss computation
    - Support for top-k distillation

    The actor works in two modes:
    1. SDPO mode (loss_mode == "sdpo"): Uses self-distillation loss
    2. Vanilla mode: Falls back to standard PPO loss

    Args:
        config: Actor configuration (should contain self_distillation section for SDPO)
        actor_module: The actor model
        actor_optimizer: Optimizer for the actor (optional, None for reference policy)
    """

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        self_distillation_cfg: Optional[Any] = None,
    ):
        super().__init__(config, actor_module, actor_optimizer)

        # SDPO-specific: Save self_distillation config (may not be in official ActorConfig)
        self._self_distillation_cfg = self_distillation_cfg or getattr(config, "self_distillation", None)
        # SDPO-specific: Teacher module for self-distillation
        self.teacher_module: Optional[nn.Module] = None

    def _update_teacher(self) -> None:
        """Update teacher module using EMA.

        This should be called after each policy update step.
        The update follows: teacher = (1 - τ) * teacher + τ * student
        """
        self_distillation_cfg = self._self_distillation_cfg
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        if not self_distillation_cfg or loss_mode != "sdpo":
            return

        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if teacher_regularization != "ema":
            return

        update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
        if update_rate == 0.0:
            return

        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("EMA teacher requires a separate teacher_module in the actor worker.")

        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_module.parameters(),
                self.actor_module.parameters(),
            ):
                student_data = student_param.data.to(device=teacher_param.device)
                teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        """Check if there are non-empty multi-modal inputs."""
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute log probability, returns dict for SDPO compatibility."""
        from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info.get("max_token_len", self.config.ppo_max_token_len_per_gpu) * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return {"log_probs": log_probs, "entropys": entropys}

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        return_all_logps=False,
        distill_topk=None,
        topk_indices=None,
        module=None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for micro batch with optional distillation support.

        This method extends the base _forward_micro_batch to support:
        - Full vocab log probabilities for distillation
        - Top-k log probabilities for memory-efficient distillation
        - Custom module (teacher/student) selection
        - rmpad and ulysses sp support for topk distillation

        Args:
            micro_batch: The micro batch data
            temperature: Sampling temperature
            calculate_entropy: Whether to calculate entropy
            return_all_logps: Whether to return full vocab log probabilities
            distill_topk: If set, return top-k log probabilities
            topk_indices: Pre-computed top-k indices (for teacher pass)
            module: Custom module to use (e.g., teacher model)

        Returns:
            Dictionary with keys: log_probs, entropys (optional),
                all_logps (optional), topk_logps (optional), topk_indices (optional)
        """
        # Use custom module if provided, otherwise use actor_module
        actor_module = module if module is not None else self.actor_module

        response_length = micro_batch["responses"].size(-1)

        # Check for multi-modal inputs
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        # Determine if we're using distillation features (which require dict return)
        use_distillation_features = (return_all_logps or distill_topk is not None or topk_indices is not None)

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            use_topk = distill_topk is not None or topk_indices is not None
            compute_all_logps = return_all_logps and not use_topk
            return_topk_indices = use_topk and topk_indices is None

            if self.use_remove_padding:
                from verl.utils.attention_utils import index_first_axis, unpad_input
                from verl.utils.ulysses import ulysses_pad, ulysses_pad_and_slice_inputs
                from einops import rearrange
                from verl.utils.attention_utils import pad_input

                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo
                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm = hasattr(
                        getattr(actor_module, "module", actor_module).config, "vision_config"
                    )
                    if is_vlm:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    if calculate_entropy:
                        entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    from verl.utils.torch_functional import logprobs_from_logits
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                    # Top-k distillation for rmpad path
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                from verl.utils.triton import slice_input_tensor
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    from verl.utils.triton import gather_outputs_and_unpad
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if topk_indices is None:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if topk_indices is None:
                            topk_indices_rmpad = topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                from einops import rearrange

                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if topk_indices is None:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                if use_topk:
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    topk_indices_out = full_topk_indices[:, -response_length - 1 : -1, :] if topk_indices is None else None
                else:
                    topk_logps = None
                    topk_indices_out = None

            else:  # not use_remove_padding
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs
                    if calculate_entropy:
                        entropy = output.entropy
                else:
                    from verl.utils.torch_functional import logprobs_from_logits
                    logits = output.logits
                    logits.div_(temperature)

                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits,
                        labels=micro_batch["responses"],
                        inplace_backward=inplace_backward,
                    )

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = self.compute_entropy_from_logits(logits)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits
                            )

                # Extract full vocab log probs if requested (non-rmpad path)
                all_logps = None
                topk_logps = None
                topk_indices_out = None
                if not self.use_fused_kernels and (return_all_logps or use_topk):
                    import torch.nn.functional as F
                    log_probs_full = F.log_softmax(logits, dim=-1)

                    if return_all_logps and not use_topk:
                        all_logps = log_probs_full

                    if use_topk:
                        if topk_indices is None:
                            topk_logps, topk_indices_out = torch.topk(log_probs_full, k=distill_topk, dim=-1)
                        else:
                            topk_logps = torch.gather(log_probs_full, -1, topk_indices.unsqueeze(-1)).squeeze(-1)
                            topk_indices_out = topk_indices

        # Return in compatible format: (entropy, log_probs) for vanilla, dict for SDPO
        if use_distillation_features:
            result = {"log_probs": log_probs}
            if calculate_entropy:
                result["entropys"] = entropy
            if return_all_logps and not use_topk:
                result["all_logps"] = all_logps
            if use_topk:
                result["topk_logps"] = topk_logps
                result["topk_indices"] = topk_indices_out if topk_indices_out is not None else topk_indices
            return result
        else:
            # Always return dict for SDPO compatibility
            result = {
                "log_probs": log_probs,
            }
            if calculate_entropy:
                result["entropys"] = entropy
            return result

    @GPUMemoryLogger(role="sdpo dp actor", logger=logger)
    def update_policy(self, data: DataProto) -> dict[str, Any]:
        """
        Update policy with SDPO support.

        This method extends the base update_policy with self-distillation
        loss computation when SDPO is enabled.

        Args:
            data: DataProto containing the batch data with keys:
                - responses: Response token ids
                - response_mask: Mask for valid response tokens
                - input_ids: Full input (prompt + response)
                - attention_mask: Attention mask
                - position_ids: Position ids
                - old_log_probs: Log probs from rollout
                - advantages: Advantage estimates
                - teacher_input_ids: (SDPO) Teacher input ids
                - teacher_attention_mask: (SDPO) Teacher attention mask
                - teacher_position_ids: (SDPO) Teacher position ids
                - self_distillation_mask: (SDPO) Mask for samples to distill

        Returns:
            Dictionary of metrics
        """
        # Make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        self_distillation_enabled = loss_mode == "sdpo"
        self_distillation_cfg = self._self_distillation_cfg

        if self_distillation_enabled:
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "self_distillation_mask",
            }
            missing_keys = self_distillation_required_keys - set(data.batch.keys())
            assert not missing_keys, f"Missing required keys for SDPO: {missing_keys}"

        # Select keys for the batch
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]

        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))

        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        # Check for multi-modal inputs
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        did_update = False

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    from verl.utils.seqlen_balancing import prepare_dynamic_batch
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = (
                        model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    )

                    if self_distillation_enabled:
                        assert not has_multi_modal_inputs, "Multi-modal inputs are not supported for distillation"

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # Determine distillation parameters
                    teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema") if self_distillation_cfg else "ema"
                    if teacher_regularization == "trust-region" and self.config.get("use_fused_kernels", False):
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")

                    return_all_logps = (
                        self_distillation_cfg.full_logit_distillation
                        and not self_distillation_cfg.distillation_topk
                    ) if self_distillation_cfg else False
                    distill_topk = (
                        self_distillation_cfg.distillation_topk
                        if self_distillation_cfg and self_distillation_cfg.full_logit_distillation
                        else None
                    )

                    # Forward pass for student
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = outputs.get("topk_indices") if distill_topk else None

                    # Get old_log_prob
                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    # Get rollout correction weights
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # Compute loss
                    if self_distillation_enabled:
                        # SDPO: Compute teacher outputs and distillation loss
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                        }
                        teacher_model = self.teacher_module or self.actor_module

                        if teacher_regularization == "trust-region" and (
                            self.teacher_module is None or self.teacher_module is self.actor_module
                        ):
                            raise ValueError(
                                "trust-region teacher requires a separate teacher_module in the actor worker."
                            )

                        with torch.no_grad():
                            teacher_outputs = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                                return_all_logps=return_all_logps,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                module=teacher_model,
                            )

                        teacher_log_prob = teacher_outputs["log_probs"]
                        teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                        teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None

                        pg_loss, pg_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                        )

                        pg_metrics["self_distillation/empty_target_batch"] = (
                            self_distillation_mask.sum().item() == 0
                        )
                        micro_batch_metrics.update(pg_metrics)
                    else:
                        # Vanilla PPO: Use standard policy loss
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Compute rollout correction metrics if available
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs
                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    # Compute total policy loss
                    policy_loss = pg_loss

                    # Add entropy bonus
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode
                        )
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    # Add KL loss if configured
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # Backward pass
                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                # Optimizer step
                grad_norm = self._optimizer_step()
                if torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()

        # Update teacher after successful gradient update
        if did_update:
            self._update_teacher()

        return metrics
