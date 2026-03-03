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
SDPO FSDP Workers.

This module extends verl's official FSDP workers to support SDPO training
by using SDPODataParallelPPOActor instead of DataParallelPPOActor.
"""

from omegaconf import OmegaConf, open_dict
from codetiming import Timer
import psutil as _psutil  # Imported for memory metrics
import torch

from verl import DataProto
from verl.single_controller.base.decorator import register, Dispatch, make_nd_compute_dataproto_dispatch_fn
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import DistProfiler
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker


class SDPOActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    SDPO-specific ActorRolloutRefWorker (synchronous version).

    This class extends the official ActorRolloutRefWorker to use
    SDPODataParallelPPOActor for SDPO training.
    """

    def __init__(self, config, role, **kwargs):
        # Save self_distillation config before parent init
        self._self_distillation_cfg = config.actor.get("self_distillation", None)
        self._sdpo_loss_mode = config.actor.policy_loss.get("loss_mode", "vanilla")

        # Temporarily remove self_distillation from config since official
        # FSDPActorConfig doesn't have this field
        if "self_distillation" in config.actor:
            OmegaConf.set_struct(config, False)
            # Keep as OmegaConf to support . access
            self._self_distillation_cfg = config.actor.self_distillation
            del config.actor.self_distillation
            OmegaConf.set_struct(config, True)

        # Call parent __init__
        super().__init__(config, role, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Override init_model to use SDPODataParallelPPOActor when SDPO mode is enabled."""
        from verl.workers.actor import DataParallelPPOActor

        # Import external libs
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            tiled_mlp_config = self.config.model.get("tiled_mlp", {})
            use_tiled_mlp = tiled_mlp_config.get("enabled", False)
            tiled_mlp_shards = tiled_mlp_config.get("num_shards", 4)

            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )

            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            # Use SDPO actor if SDPO mode, otherwise standard actor
            if self._sdpo_loss_mode == "sdpo":
                from recipe.sdpo.dp_actor import SDPODataParallelPPOActor
                self.actor = SDPODataParallelPPOActor(
                    config=actor_cfg,
                    actor_module=self.actor_module_fsdp,
                    actor_optimizer=self.actor_optimizer,
                    self_distillation_cfg=self._self_distillation_cfg,
                )
            else:
                self.actor = DataParallelPPOActor(
                    config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
                )
            if getattr(self, "tokenizer", None) is not None:
                self.actor.tokenizer = self.tokenizer

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)

            ref_tiled_mlp_config = self.config.ref.get("tiled_mlp", None)
            if ref_tiled_mlp_config is None:
                ref_tiled_mlp_config = self.config.model.get("tiled_mlp", {})
            ref_use_tiled_mlp = ref_tiled_mlp_config.get("enabled", False)
            ref_tiled_mlp_shards = ref_tiled_mlp_config.get("num_shards", 4)

            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
                use_tiled_mlp=ref_use_tiled_mlp,
                tiled_mlp_shards=ref_tiled_mlp_shards,
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
            from verl.utils.flops_counter import FlopsCounter

            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

        # SDPO: Set up teacher_module after actor is created
        if self._is_actor and self._sdpo_loss_mode == "sdpo" and self._self_distillation_cfg is not None:
            teacher_regularization = self._self_distillation_cfg.get("teacher_regularization", "ema")
            if teacher_regularization == "trust-region":
                from recipe.sdpo.dp_actor import TrustRegionTeacher
                self.actor.teacher_module = TrustRegionTeacher(
                    ref_module=self.ref_module_fsdp,
                    student_module=self.actor_module_fsdp,
                    mix_coef=self._self_distillation_cfg.get("teacher_update_rate", 0.0),
                )
            else:
                self.actor.teacher_module = self.ref_module_fsdp

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        """Override update_actor to use ulysses_sharding_manager."""
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)

            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time
            )
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            from verl.utils.device import get_torch_device
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = _psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            # Return DataProto with metrics (following official verl)
            output = DataProto(meta_info={"metrics": metrics})

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        """Override compute_log_prob to handle dict return from SDPO actor."""
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()

        # Use ref or rollout config based on is_lora
        config_source = self.config.ref if is_lora else self.config.rollout
        data.meta_info["micro_batch_size"] = config_source.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = config_source.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = config_source.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        calculate_entropy = not is_lora
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                outputs = self.actor.compute_log_prob(data=data, calculate_entropy=calculate_entropy)

        if not is_lora:
            tensors = {"old_log_probs": outputs["log_probs"]}
        else:
            tensors = {"ref_log_prob": outputs["log_probs"]}
        if calculate_entropy:
            tensors["entropys"] = outputs["entropys"]
        if "sum_pi_squared" in outputs:
            tensors["sum_pi_squared"] = outputs["sum_pi_squared"]

        output = DataProto.from_dict(
            tensors=tensors,
            meta_info={"temperature": self.config.rollout.temperature},
        )

        output = output.to("cpu")

        # Unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        return output


class AsyncSDPOActorRolloutRefWorker(SDPOActorRolloutRefWorker, AsyncActorRolloutRefWorker):
    """
    SDPO-specific AsyncActorRolloutRefWorker (asynchronous version).

    This class extends AsyncActorRolloutRefWorker to use SDPODataParallelPPOActor
    for SDPO training. The SDPO features are inherited from SDPOActorRolloutRefWorker.

    All methods are inherited from parent classes:
    - __init__, _setup_teacher_module from SDPOActorRolloutRefWorker
    - wake_up, sleep, get_zeromq_address from AsyncActorRolloutRefWorker
    """
