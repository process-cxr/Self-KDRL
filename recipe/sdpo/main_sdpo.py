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
SDPO Main Entry Point.

This is the main entry point for SDPO training, following verl's recipe pattern.
The key difference from vanilla PPO is that SDPO uses self-distillation loss
instead of the standard PPO clipped loss.

Usage:
    python -m recipe.sdpo.main_sdpo --config-name sdpo_trainer \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
        data.train_files=datasets/tooluse/train.parquet
"""

import logging
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.device import auto_set_device, is_cuda_available

from sdpo_trainer import RaySDPOTrainer

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="sdpo_trainer", version_base=None)
def main(config):
    """Main entry point for SDPO training."""
    # Automatically set device (NPU or CUDA)
    auto_set_device(config)
    run_sdpo(config)


def run_sdpo(config) -> None:
    """Initialize Ray cluster and run distributed SDPO training."""
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_kwargs", {}).get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"Ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        # Check for profiler settings
        if (
            is_cuda_available
            and config.get("global_profiler", {}).get("tool") == "nsys"
            and OmegaConf.select(config, "global_profiler.steps") is not None
            and len(OmegaConf.select(config, "global_profiler.steps")) > 0
        ):
            nsight_options = OmegaConf.to_container(
                config.global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)
class TaskRunner:
    """Ray remote class for executing SDPO training."""

    def run(self, config):
        """Run SDPO training."""
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        # Print and resolve config
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download checkpoint if needed
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # Initialize tokenizer
        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)

        from verl.single_controller.ray import RayWorkerGroup

        # SDPO configuration validation
        self_distillation_cfg = config.actor_rollout_ref.get("self_distillation", None)
        loss_mode = config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        self_distillation_needs_ref = self_distillation_cfg is not None and loss_mode == "sdpo"

        # SDPO requires reference policy for self-distillation
        # Note: role_worker_mapping is not available yet, so we skip this check here

        # Define worker classes based on strategy
        # For SDPO, we use custom workers that support self-distillation
        # Note: We use AsyncActorRolloutRefWorker for async rollout (default mode)
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from recipe.sdpo.fsdp_workers import AsyncSDPOActorRolloutRefWorker as ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from recipe.sdpo.fsdp_workers import AsyncSDPOActorRolloutRefWorker as ActorRolloutRefWorker
            # TODO: Verify megatron SDPO worker implementation
            logger.warning("Megatron SDPO worker not fully tested")
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(
                f"Strategy {config.actor_rollout_ref.actor.strategy} not supported"
            )

        # Build role-worker mapping
        from verl.trainer.ppo.utils import Role
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        # Add reward model if enabled
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError(
                    f"Reward model strategy {config.reward_model.strategy} not supported"
                )
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add reference policy if needed
        if need_reference_policy(role_worker_mapping):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Add critic if using GAE
        if config.algorithm.adv_estimator == "gae":
            if config.critic.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import CriticWorker
            elif config.critic.strategy == "megatron":
                from verl.workers.megatron_workers import CriticWorker
            else:
                raise NotImplementedError(
                    f"Critic strategy {config.critic.strategy} not supported"
                )
            role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
            mapping[Role.Critic] = global_pool_id

        # Validate config
        from verl.utils.config import validate_config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(role_worker_mapping),
            use_critic=config.algorithm.adv_estimator == "gae",
        )

        # Load reward manager
        reward_fn = load_reward_manager(
            config=config,
            tokenizer=tokenizer,
            num_examine=0,
        )
        val_reward_fn = load_reward_manager(
            config=config,
            tokenizer=tokenizer,
            num_examine=1,
        )

        # Create resource pool manager
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping
        )

        # Create SDPO trainer
        # Note: RaySDPOTrainer inherits from RayPPOTrainer and overrides
        # _maybe_build_self_distillation_batch() to build teacher inputs
        trainer = RaySDPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
