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
SDPO Ray Trainer.

This module provides the SDPO-specific trainer implementation that extends
verl's RayPPOTrainer with self-distillation capabilities.
"""

import logging
import re
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask

from .config import SelfDistillationConfig

logger = logging.getLogger(__name__)


class RaySDPOTrainer(RayPPOTrainer):
    """
    SDPO-specific Ray Trainer.

    This class extends verl's RayPPOTrainer with self-distillation
    capabilities for SDPO training.

    Key additions:
    - Building self-distillation batch from high-reward trajectories
    - Collecting environment feedback
    - Managing teacher inputs
    """

    def __init__(
        self,
        config,
        tokenizer: PreTrainedTokenizer,
        processor=None,
        role_worker_mapping=None,
        resource_pool_manager=None,
        ray_worker_group_cls=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        # SDPO-specific initialization
        self._init_sdpo_config()

    def _init_sdpo_config(self):
        """Initialize SDPO-specific configuration."""
        self.self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        self.loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")

        if self.self_distillation_cfg is not None and self.loss_mode == "sdpo":
            self._sdpo_enabled = True
            self.tokenizer.truncation_side = getattr(
                self.self_distillation_cfg, "reprompt_truncation", "error"
            )
        else:
            self._sdpo_enabled = False

    def _collect_solutions_by_uid(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        success_reward_threshold: float,
    ) -> dict[Any, list[int]]:
        """Collect successful solution indices grouped by UID."""
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]

        success_by_uid: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            if seq_scores[idx] >= success_reward_threshold:
                success_by_uid[uid].append(idx)

        return success_by_uid

    @staticmethod
    def _collect_feedback(
        include_environment_feedback: bool,
        reward_extra_infos_dict: Optional[dict[str, Any]],
        batch_size: int,
    ) -> list[Any]:
        """Collect environment feedback from reward extra info."""
        feedback_list: list[Any] = [None] * batch_size

        if include_environment_feedback and reward_extra_infos_dict is not None:
            raw_feedback = reward_extra_infos_dict.get("feedback", [])
            for i in range(min(len(raw_feedback), batch_size)):
                if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                    feedback_list[i] = raw_feedback[i]

        return feedback_list

    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        """Remove thinking trace tags from text."""
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

    def _get_solution(
        self,
        idx: int,
        success_by_uid: dict[Any, list[int]],
        uids: list[Any],
        response_texts: list[str],
        dont_reprompt_on_self_success: bool = False,
        remove_thinking_from_demonstration: bool = False,
    ) -> Optional[str]:
        """Get a successful solution for the given index."""
        uid = uids[idx]
        solution_idxs = success_by_uid[uid]

        if dont_reprompt_on_self_success:
            solution_idxs = [j for j in solution_idxs if j != idx]

        if len(solution_idxs) == 0:
            return None

        # Take the first successful demonstration (effectively random)
        solution_idx = solution_idxs[0]
        solution_str = response_texts[solution_idx]

        if remove_thinking_from_demonstration:
            solution_str = self._remove_thinking_trace(solution_str)

        return solution_str

    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        """
        Build self-distillation batch for SDPO training.

        This method constructs teacher inputs by combining:
        - Original prompt
        - Successful solutions from the same prompt group
        - Environment feedback (if available)

        Args:
            batch: DataProto containing the batch
            reward_tensor: Tensor of rewards
            reward_extra_infos_dict: Dictionary containing extra reward info

        Returns:
            Tuple of (DataProto with teacher inputs, metrics dict) or None
        """
        if not self._sdpo_enabled:
            return None

        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch["raw_prompt"]]
        batch_size = batch.batch.batch_size[0]

        # Collect feedback
        feedback_list = self._collect_feedback(
            include_environment_feedback=self.self_distillation_cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )

        # Collect successful solutions
        success_by_uid = self._collect_solutions_by_uid(
            batch, reward_tensor,
            success_reward_threshold=self.self_distillation_cfg.success_reward_threshold
        )

        # Get solutions for each sample
        solution_strs = [
            self._get_solution(
                i, success_by_uid, batch.non_tensor_batch["uid"], response_texts,
                self.self_distillation_cfg.dont_reprompt_on_self_success,
                self.self_distillation_cfg.get("remove_thinking_from_demonstration", False),
            )
            for i in range(batch_size)
        ]

        # Build teacher messages
        def _build_teacher_message(i: int) -> list[dict]:
            system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
            has_solution = solution_strs[i] is not None
            has_feedback = feedback_list[i] is not None
            feedback_only_without_solution = self.self_distillation_cfg.get(
                "environment_feedback_only_without_solution", False
            )

            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

            # Build solution section
            solution_section = ""
            if has_solution:
                solution_section = self.self_distillation_cfg.solution_template.format(
                    successful_previous_attempt=solution_strs[i]
                )

            # Build feedback section
            feedback_section = ""
            if use_feedback:
                feedback_section = self.self_distillation_cfg.feedback_template.format(
                    feedback_raw=feedback_list[i]
                )

            # Combine
            if use_feedback or has_solution:
                reprompt_text = self.self_distillation_cfg.reprompt_template.format(
                    prompt=prompt_texts[i],
                    solution=solution_section,
                    feedback=feedback_section,
                )
            else:
                reprompt_text = prompt_texts[i]

            return system_messages + [{"role": "user", "content": reprompt_text}]

        # Build teacher prompts
        teacher_messages = [_build_teacher_message(i) for i in range(batch_size)]

        enable_thinking = batch.meta_info.get("enable_thinking", False)
        teacher_prompt = self.tokenizer.apply_chat_template(
            teacher_messages,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            max_length=self.self_distillation_cfg.max_reprompt_len,
            padding=True,
            truncation=True,
        )

        # Concatenate teacher prompt with responses
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)

        # Compute position ids
        from verl.utils.model import compute_position_id_with_mask
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        # Compute self_distillation_mask
        feedback_only_without_solution = self.self_distillation_cfg.get(
            "environment_feedback_only_without_solution", False
        )
        feedback_used = [
            feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
            for i in range(batch_size)
        ]
        self_distillation_mask = torch.tensor(
            [solution_strs[i] is not None or feedback_used[i] for i in range(batch_size)],
            dtype=torch.float32,
            device=device
        )

        # Compute metrics
        uids = set(batch.non_tensor_batch["uid"])
        num_with_feedback_available = sum(1 for f in feedback_list if f is not None)
        num_with_feedback_used = sum(1 for f in feedback_used if f)
        num_with_solution = sum(1 for s in solution_strs if s is not None)

        metrics = {
            "self_distillation/success_group_fraction": len([uid for uid in uids if len(success_by_uid[uid]) > 0]) / len(uids),
            "self_distillation/success_sample_fraction": num_with_solution / batch_size,
            "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
            "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
            "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
        }

        return DataProto.from_dict(tensors={
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_position_ids": teacher_position_ids,
            "self_distillation_mask": self_distillation_mask,
        }), metrics
