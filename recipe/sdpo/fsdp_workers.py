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

import logging

from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

logger = logging.getLogger(__name__)


class SDPOActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    SDPO-specific ActorRolloutRefWorker (synchronous version).

    This class extends the official ActorRolloutRefWorker to use
    SDPODataParallelPPOActor for SDPO training. The key differences:

    1. Uses SDPODataParallelPPOActor which supports:
       - teacher_module for self-distillation
       - extended _forward_micro_batch for distillation
       - SDPO-specific update_policy method

    2. Sets up teacher_module based on self_distillation config:
       - 'ema': Uses ref_module_fsdp as teacher
       - 'trust-region': Creates TrustRegionTeacher wrapper
       - default: Uses actor as teacher (teacher_module=None)
    """

    def __init__(self, config, role, **kwargs):
        # First call parent __init__ to create standard actor
        super().__init__(config, role, **kwargs)

        # Then replace actor with SDPODataParallelPPOActor if SDPO is enabled
        self._maybe_replace_actor_with_sdpo()

    def _maybe_replace_actor_with_sdpo(self):
        """Replace the actor with SDPODataParallelPPOActor if SDPO is enabled."""
        # Check if SDPO is enabled
        loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "sdpo":
            return

        # Only replace if we have an actor (not rollout-only)
        if not hasattr(self, "actor"):
            return

        from recipe.sdpo.dp_actor import SDPODataParallelPPOActor

        # Create SDPO actor with same config
        actor_cfg = self.actor.config
        actor_module = self.actor.actor_module
        actor_optimizer = self.actor.actor_optimizer

        self.actor = SDPODataParallelPPOActor(
            config=actor_cfg, actor_module=actor_module, actor_optimizer=actor_optimizer
        )

        # Set up teacher_module
        self._setup_teacher_module()

    def _setup_teacher_module(self):
        """Set up teacher_module for SDPO training.

        This method configures the teacher module based on the
        self_distillation.teacher_regularization setting:
        - 'ema': Use ref_module_fsdp as separate teacher
        - 'trust-region': Creates TrustRegionTeacher wrapper
        - default: Use actor as teacher (teacher_module=None)
        """
        # Check if SDPO is enabled
        loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "sdpo":
            return

        self_distillation_cfg = self.config.actor.get("self_distillation", None)
        if self_distillation_cfg is None:
            logger.warning("SDPO enabled but self_distillation config is missing")
            return

        teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
        logger.info(f"Setting up SDPO teacher_module with mode: {teacher_regularization}")

        if teacher_regularization == "ema":
            # For EMA, we need a separate teacher module
            # Use ref_module_fsdp if available (it's already a separate model)
            if hasattr(self, "ref_module_fsdp") and self.ref_module_fsdp is not None:
                self.actor.set_teacher_module(self.ref_module_fsdp)
                logger.info("  Using ref_module_fsdp as teacher (EMA mode)")
            else:
                logger.warning(
                    "  No ref_module_fsdp available for EMA teacher. "
                    "Actor will be used as teacher (not recommended for EMA)."
                )

        elif teacher_regularization == "trust-region":
            # For trust-region, create TrustRegionTeacher that combines ref and student
            if hasattr(self, "ref_module_fsdp") and self.ref_module_fsdp is not None:
                from recipe.sdpo.dp_actor import TrustRegionTeacher

                # Get the FSDP-wrapped modules (unwrap to get the base module)
                if hasattr(self.ref_module_fsdp, "_fsdp_wrapped_module"):
                    ref_module = self.ref_module_fsdp._fsdp_wrapped_module
                else:
                    ref_module = self.ref_module_fsdp

                if hasattr(self.actor.actor_module, "_fsdp_wrapped_module"):
                    student_module = self.actor.actor_module._fsdp_wrapped_module
                else:
                    student_module = self.actor.actor_module

                self.actor.set_teacher_module(
                    TrustRegionTeacher(
                        ref_module=ref_module,
                        student_module=student_module,
                        mix_coef=self_distillation_cfg.get("teacher_update_rate", 0.0),
                    )
                )
                logger.info("  Using TrustRegionTeacher wrapper")
            else:
                logger.warning("  No ref_module_fsdp available for trust-region teacher")

        # For default (no teacher_regularization), teacher_module remains None
        # and actor will use itself as teacher


class AsyncSDPOActorRolloutRefWorker(SDPOActorRolloutRefWorker, AsyncActorRolloutRefWorker):
    """
    SDPO-specific AsyncActorRolloutRefWorker (asynchronous version).

    This class extends AsyncActorRolloutRefWorker to use SDPODataParallelPPOActor
    for SDPO training. The SDPO features are inherited from SDPOActorRolloutRefWorker.

    All methods are inherited from parent classes:
    - __init__, _setup_teacher_module from SDPOActorRolloutRefWorker
    - wake_up, sleep, get_zeromq_address from AsyncActorRolloutRefWorker
    """
