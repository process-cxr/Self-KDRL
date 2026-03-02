# SDPO 训练数据特点与处理指南

## 一、SDPO 数据特点

### 1.1 与普通 RLHF 数据的对比

| 特点 | 普通 RLHF | SDPO |
|------|----------|------|
| **奖励来源** | 预训练的 Reward Model | 可验证奖励（代码测试、数学验证等） |
| **反馈类型** | 标量分数 | 丰富的文本反馈（可选） |
| **数据要求** | prompt + response + reward | prompt + ground_truth/tests |
| **成功轨迹** | 不需要 | 用于构建 teacher 输入 |

### 1.2 SDPO 数据核心要求

```json
{
  "prompt": "问题文本",
  "answer": "标准答案",
  "tests": "测试用例（代码任务）",
  "kind": "math|code|...",
  "dataset": "数据集来源",
  "elo": "难度分数"
}
```

**关键字段说明**：

| 字段 | 类型 | 用途 |
|------|------|------|
| `prompt` | string | 问题文本，作为 student 输入 |
| `answer` | string | 标准答案，用于奖励计算 |
| `tests` | string | 测试用例（代码任务），用于生成反馈 |
| `kind` | string | 决定使用哪个奖励函数 (`code`, `math`, etc.) |
| `dataset` | string | 数据来源标识，决定 `data_source` |
| `elo` | string | 难度分数，可用于课程学习 |

---

## 二、数据处理流程

### 2.1 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SDPO 数据处理流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 下载/加载数据                                                      │
│  ┌─────────────┐                                                            │
│  │ load_dataset│  → 生成 train.json / test.json                            │
│  │    .py      │                                                            │
│  └─────────────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  Step 2: 分割数据集（可选）                                                  │
│  ┌─────────────┐     ┌─────────────┐                                        │
│  │split_tests  │ 或  │split_tasks  │                                        │
│  │    .py      │     │    .py      │                                        │
│  └─────────────┘     └─────────────┘                                        │
│         │                                                                   │
│         ▼                                                                   │
│  Step 3: 预处理为 parquet                                                   │
│  ┌─────────────┐                                                            │
│  │ preprocess  │  → 生成 train.parquet / test.parquet                       │
│  │    .py      │                                                            │
│  └─────────────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  Step 4: 训练时加载                                                         │
│  ┌─────────────┐                                                            │
│  │rl_dataset.py│  → RLHFDataset                                             │
│  └─────────────┘                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 各步骤详解

#### Step 1: 下载/加载数据

```bash
# LiveCodeBench v6 (代码任务)
python data/load_dataset.py \
    --dataset_name livecodebench/code_generation_lite-v6 \
    --output_path datasets/lcb_v6.json

# SciKnowEval (科学知识)
python data/load_dataset.py \
    --dataset_name Chemistry \
    --output_path datasets/chemistry.json

# ToolUse (已内置)
# datasets/tooluse/train.json 和 test.json 已存在
```

**输出格式**：
```json
{
  "idx": "unique_id",
  "prompt": "问题文本",
  "answer": "标准答案",
  "tests": "测试用例",
  "description": "问题描述",
  "kind": "code",
  "dataset": "livecodebench",
  "elo": "1500"
}
```

#### Step 2: 分割数据集

```bash
# LiveCodeBench - 按测试用例分割
python data/split_tests.py \
    --json_path datasets/lcb_v6.json \
    --output_dir datasets/lcb_v6

# SciKnowEval - 按任务分割
python data/split_tasks.py \
    --json_path datasets/chemistry.json \
    --output_dir datasets/sciknoweval/chemistry
```

#### Step 3: 预处理为 parquet

```bash
python data/preprocess.py --data_source datasets/tooluse
```

**转换后的格式**：
```json
{
  "data_source": "tooluse",
  "prompt": [
    {"role": "user", "content": "问题文本"}
  ],
  "ability": "code",
  "reward_model": {
    "style": "code",
    "ground_truth": "测试用例或答案"
  },
  "extra_info": {
    "split": "train",
    "index": "xxx",
    "description": "...",
    "problem": "...",
    "elo": "..."
  }
}
```

---

## 三、数据字段映射

### 3.1 原始数据 → 预处理后数据

```
原始字段                  预处理后字段
───────────────────────────────────────────────
prompt            →    prompt (messages格式)
answer            →    reward_model.ground_truth
tests             →    reward_model.ground_truth (code任务)
kind              →    ability, reward_model.style
dataset           →    data_source
description       →    extra_info.description
elo               →    extra_info.elo
idx               →    extra_info.index
```

### 3.2 训练时数据流

```
parquet 文件
     │
     ▼
RLHFDataset.__getitem__()
     │
     ├── tokenization
     │
     ▼
DataProto {
    batch: {
        "input_ids": tensor,
        "attention_mask": tensor,
        "position_ids": tensor,
    },
    non_tensor_batch: {
        "raw_prompt": messages,
        "data_source": string,
        "reward_model": dict,
        "extra_info": dict,
    }
}
     │
     ▼
Rollout 生成 n 个响应
     │
     ▼
奖励计算 (根据 data_source 选择奖励函数)
     │
     ├── code → 执行测试用例 → 奖励 + 反馈
     ├── math → 数学验证 → 奖励
     └── ...
     │
     ▼
_build_self_distillation_batch()
     │
     ├── 收集成功轨迹
     ├── 收集环境反馈
     └── 构建 teacher_input_ids
```

---

## 四、支持的数据集类型

### 4.1 代码任务（支持环境反馈）

| 数据集 | kind | 反馈类型 |
|--------|------|----------|
| LiveCodeBench | code | 测试用例执行结果 |
| Codeforces | code | 测试用例执行结果 |
| ToolUse | code | 工具调用验证 |

**反馈示例**：
```
Test Case 1:
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
Expected: [0, 1]  ✓

Test Case 2:
Input: nums = [3, 2, 4], target = 6
Output: [1, 2]
Expected: [0, 2]  ✗

Runtime Error on Test Case 3:
NameError: name 'Solution' is not defined
```

### 4.2 数学任务

| 数据集 | kind | 验证方式 |
|--------|------|----------|
| MATH | math | 答案匹配 |
| GSM8K | math | 答案匹配 |
| AIME | math | 答案匹配 |

### 4.3 科学知识

| 数据集 | kind | 验证方式 |
|--------|------|----------|
| Biology | sciknoweval | 选择题匹配 |
| Chemistry | sciknoweval | 选择题匹配 |
| Material | sciknoweval | 选择题匹配 |
| Physics | sciknoweval | 选择题匹配 |

---

## 五、自定义数据集

### 5.1 添加新数据集

1. **创建加载函数**（在 `data/format/` 目录）：

```python
# data/format/my_dataset.py
from datasets import load_dataset

def load_my_dataset():
    ds = load_dataset("my_dataset_name")
    # 统一字段名
    ds = ds.rename_column("question", "prompt")
    ds = ds.rename_column("solution", "answer")
    ds = ds.add_column("kind", ["code"] * len(ds))
    ds = ds.add_column("dataset", ["my_dataset"] * len(ds))
    return ds
```

2. **注册到 load_dataset.py**：

```python
# data/load_dataset.py
from data.format.my_dataset import load_my_dataset

def load_dataset_hf(...):
    ...
    elif dataset_name == "my_dataset":
        ds = load_my_dataset()
```

3. **添加奖励函数**（如果需要）：

```python
# verl/utils/reward_score/my_reward.py
def compute_score(solution, ground_truth, extra_info=None):
    # 自定义奖励逻辑
    return {"score": reward, "feedback": feedback_text}
```

### 5.2 数据格式要求

**最小必填字段**：

```json
{
  "prompt": "问题文本",
  "answer": "标准答案（用于验证）",
  "kind": "任务类型",
  "dataset": "数据集名称"
}
```

**可选字段**：

```json
{
  "tests": "测试用例（代码任务必填）",
  "description": "问题描述",
  "elo": "难度分数",
  "system": "系统提示词"
}
```

---

## 六、实际示例

### 6.1 ToolUse 数据集样例

```json
{
  "prompt": "Your task is to answer the user's question using available tools...\nQuestion: I'm troubleshooting some requests...",
  "description": "Your task is to answer the user's question...",
  "tests": "[{\"Action\": \"sendHttpRequest\", \"Action_Input\": {...}}]",
  "kind": "code",
  "dataset": "tooluse",
  "elo": "1200"
}
```

### 6.2 LiveCodeBench 数据集样例

```json
{
  "prompt": "Write a function to find two numbers that add up to target...",
  "description": "You are given an array of integers...",
  "tests": {
    "inputs": ["nums = [2,7,11,15], target = 9", ...],
    "outputs": ["[0,1]", ...]
  },
  "kind": "code",
  "dataset": "livecodebench",
  "elo": "1500"
}
```

### 6.3 化学数据集样例

```json
{
  "prompt": "Which of the following compounds is most acidic?",
  "answer": "C",
  "tests": "-",
  "description": "Organic chemistry question...",
  "kind": "math",
  "dataset": "chemistry",
  "elo": "1800"
}
```

---

## 七、SDPO 特有数据流

### 7.1 Rollout 阶段

```python
# 每个 prompt 生成 n 个响应
for prompt in batch:
    responses = model.generate(prompt, n=8)
    for response in responses:
        # 计算奖励
        result = compute_score(
            solution=response,
            ground_truth=ground_truth,
            extra_info=extra_info
        )
        # result = {"score": 0.75, "feedback": "Test Case 1 passed..."}
```

### 7.2 构建 Teacher 输入

```python
# _maybe_build_self_distillation_batch
for sample in batch:
    if sample.reward >= success_threshold:
        # 收集成功轨迹
        success_solutions[uid].append(sample.response)

for sample in batch:
    if sample.reward < success_threshold:
        # 为失败样本构建 teacher 输入
        teacher_prompt = f"""
        {original_prompt}

        Correct solution:
        {success_solutions[uid][0]}

        Feedback from your attempt:
        {sample.feedback}

        Correctly solve the original question.
        """
```

---

## 八、常见问题

### Q1: 如何处理没有测试用例的任务？

A: 对于数学等任务，`tests` 字段可以为空或不提供，奖励函数会使用 `answer` 字段进行验证。

### Q2: 反馈文本太长怎么办？

A: 配置 `max_reprompt_len` 限制 teacher prompt 长度：
```yaml
self_distillation:
  max_reprompt_len: 10240
  reprompt_truncation: right
```

### Q3: 如何禁用环境反馈？

A: 配置：
```yaml
self_distillation:
  include_environment_feedback: false
```

### Q4: 数据量不足怎么办？

A: SDPO 支持从少量成功轨迹中学习。建议：
- 增加 `rollout.n`（每个 prompt 的响应数）
- 降低 `success_reward_threshold`
