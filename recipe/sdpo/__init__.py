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
SDPO (Self-Distilled Policy Optimization) Recipe for verl.

This recipe implements SDPO by extending verl's base classes without modifying
the original verl source code.

Usage:
    python -m recipe.sdpo.main_sdpo --config-name sdpo_trainer
"""

from .core_algos import compute_self_distillation_loss
from .dp_actor import SDPODataParallelPPOActor, TrustRegionTeacher
from .sdpo_trainer import RaySDPOTrainer
from .config import SelfDistillationConfig

__all__ = [
    "compute_self_distillation_loss",
    "SDPODataParallelPPOActor",
    "TrustRegionTeacher",
    "RaySDPOTrainer",
    "SelfDistillationConfig",
]
