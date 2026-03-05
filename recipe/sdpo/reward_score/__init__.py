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
SDPO Reward Score Module.

This module provides reward computation functions with environment feedback
for SDPO training. It automatically selects the appropriate reward function
based on the data_source parameter.
"""

import sys
import os
from typing import Optional

# Add reward_score directory to path for absolute imports
if __name__ != "__main__":
    reward_score_dir = os.path.dirname(os.path.abspath(__file__))
    if reward_score_dir not in sys.path:
        sys.path.insert(0, reward_score_dir)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> dict:
    """Compute reward score based on data_source.

    This function acts as a dispatcher, automatically selecting the appropriate
    reward computation function based on the data_source parameter.

    Args:
        data_source: The type of dataset/task (e.g., "tooluse", "code", "math", "gpqa")
        solution_str: The model's response
        ground_truth: The ground truth answer
        extra_info: Additional information for reward computation

    Returns:
        dict: Dictionary containing score and optional metadata
    """
    if data_source in ["code", "livecodebench", "humanevalplus", "codeforces", "code_elo", "mbppplus"]:
        # Import dynamically to avoid loading all modules
        import code
        return code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    elif data_source in ["math", "math500", "dapo_math", "gsm8k", "aime24", "aime25", "amc23"]:
        import math
        return math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["gpqa"]:
        import gpqa
        return gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval"]:
        import mcq
        return mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["tooluse"]:
        import tooluse
        return tooluse.compute_score(solution_str, ground_truth)
    elif data_source in ["mmlu_pro"]:
        import mmlu_pro
        return mmlu_pro.compute_score(solution_str, ground_truth)
    else:
        raise ValueError(f"Reward style {data_source} not found. Available: code, livecodebench, humanevalplus, codeforces, code_elo, mbppplus, math, math500, dapo_math, gsm8k, aime24, aime25, amc23, gpqa, sciknoweval, tooluse, mmlu_pro")


# Backward compatibility: expose individual functions
__all__ = [
    "compute_score",
]
