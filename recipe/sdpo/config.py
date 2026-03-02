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
SDPO Configuration Classes.

This module defines configuration dataclasses for SDPO.
"""

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for self-distillation loss.

    Distillation is enabled when policy_loss.loss_mode == "sdpo".

    Args:
        full_logit_distillation: Whether to use full-logit KL distillation.
        alpha: KL interpolation coefficient. 0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        success_reward_threshold: Minimum sequence reward to be considered successful.
        teacher_regularization: Teacher regularization mode. Options: "ema", "trust-region".
        teacher_update_rate: EMA update rate for teacher weights, or trust-region mixing coefficient.
        distillation_topk: If set, use top-k logits for distillation (saves memory).
        distillation_add_tail: Whether to add a tail bucket for top-k distillation.
        max_reprompt_len: Maximum length of the reprompted prompt.
        reprompt_truncation: Truncation method for the reprompted prompt.
        dont_reprompt_on_self_success: Whether to not reprompt on self-success.
        remove_thinking_from_demonstration: Whether to remove <think>...</think> tags from demonstrations.
        is_clip: Clip value for distillation IS ratio; None disables IS weighting.
        reprompt_template: Template for reprompting. Uses {prompt}, {solution}, {feedback} placeholders.
        solution_template: Template for formatting solution section.
        feedback_template: Template for formatting feedback section.
        include_environment_feedback: Whether to include environment feedback in reprompting.
        environment_feedback_only_without_solution: If True, only use feedback when no solution is available.
    """

    full_logit_distillation: bool = True
    alpha: float = 0.5
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None

    # Templates for reprompting
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )

    # Environment feedback settings
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")

        valid_teacher_regularization = ["ema", "trust-region"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )

        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )

        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )

        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
