# SDPO Recipe 开发记录

## 项目背景

将 Self-KDRL 项目中的 SDPO 算法从 verl 源码中剥离出来，按照 verl 的 recipe 设计模式进行重构，实现代码隔离。

## 原始问题

1. Self-KDRL 项目基于 verl 改造，但直接修改了 verl 源码
2. 删除了上游 verl 的 commit 历史，难以追踪改动
3. 代码耦合严重，难以维护和升级

## 解决方案

参照 verl 官方 recipe 设计模式（如 `recipe/spin/`、`recipe/dapo/`），创建独立的 `recipe/sdpo/` 目录，通过继承的方式实现 SDPO 功能。

---

## 关键发现

### 1. 上游版本确认

通过代码中的线索确定上游 verl 版本：
- `verl/version/version`: `0.7.0.dev`
- `docker/Dockerfile.stable.sglang`: `verl@v0.6.0`

### 2. SDPO 核心改动文件

对比原始 verl 后，SDPO 主要修改了以下文件：

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `verl/trainer/ppo/core_algos.py` | 新增函数 | `compute_self_distillation_loss()` |
| `verl/workers/actor/dp_actor.py` | 新增类+修改 | `TrustRegionTeacher`, `_update_teacher()`, `update_policy()` |
| `verl/trainer/ppo/ray_trainer.py` | 新增方法 | `_maybe_build_self_distillation_batch()` |
| `verl/trainer/config/sdpo.yaml` | 新增配置 | SDPO 配置文件 |
| `verl/utils/reward_score/feedback/` | 新增目录 | 带反馈的奖励函数 |

### 3. SDPO 算法核心

```
┌─────────────────────────────────────────────────────────────────┐
│                      SDPO 训练流程                               │
├─────────────────────────────────────────────────────────────────┤
│  1. Rollout: 生成 n 个响应 → 计算奖励                            │
│  2. 筛选: 找出高奖励轨迹 → 构建 teacher prompt                    │
│  3. Distill: Student vs Teacher → JSD Loss                      │
│  4. Update: EMA 更新教师模型                                     │
└─────────────────────────────────────────────────────────────────┘
```

**关键参数**：
- `alpha=0.5`: JSD (Jensen-Shannon Divergence)
- `teacher_regularization="ema"`: EMA 教师更新
- `teacher_update_rate=0.05`: EMA 更新率

---

## 创建的 Recipe 结构

```
recipe/sdpo/
├── __init__.py              # 模块导出
├── main_sdpo.py             # 入口脚本 (Hydra + Ray)
├── sdpo_trainer.py          # RaySDPOTrainer (继承 RayPPOTrainer)
│   └── 覆盖: _maybe_build_self_distillation_batch()
├── dp_actor.py              # SDPODataParallelPPOActor (继承 DataParallelPPOActor)
│   └── 覆盖: update_policy()
│   └── 新增: _update_teacher(), set_teacher_module()
├── core_algos.py            # compute_self_distillation_loss()
├── config.py                # SelfDistillationConfig dataclass
├── config/
│   └── sdpo_trainer.yaml    # SDPO 配置
├── reward_score/
│   ├── __init__.py
│   ├── code.py              # 代码奖励 + LeetCode 风格反馈
│   └── tooluse.py           # ToolUse 奖励 + 反馈
├── run_sdpo.sh              # 启动脚本
└── README.md                # 详细文档
```

---

## 使用方式

```bash
# 方式一：使用启动脚本
bash recipe/sdpo/run_sdpo.sh my_experiment

# 方式二：直接运行 Python
python -m recipe.sdpo.main_sdpo \
    --config-name sdpo_trainer \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    data.train_files=datasets/tooluse/train.parquet
```

---

## 关键代码片段

### 1. Self-Distillation Loss (JSD)

```python
# JSD = (1-α) * KL(M || student) + α * KL(M || teacher)
# M = (1-α) * student + α * teacher

alpha = 0.5
mixture_log_probs = torch.logsumexp([
    student_log_probs + log(1-alpha),
    teacher_log_probs + log(alpha)
], dim=0)

kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs)
kl_student = F.kl_div(mixture_log_probs, student_log_probs)
loss = torch.lerp(kl_student, kl_teacher, alpha)
```

### 2. EMA Teacher Update

```python
def _update_teacher(self):
    # teacher = (1 - τ) * teacher + τ * student
    for teacher_param, student_param in zip(...):
        teacher_param.data.mul_(1 - tau).add_(student_data, alpha=tau)
```

### 3. 构建 Teacher Input

```python
def _maybe_build_self_distillation_batch(self, batch, reward_tensor, ...):
    # 1. 收集成功轨迹
    success_by_uid = self._collect_solutions_by_uid(batch, reward_tensor, threshold)

    # 2. 构建 teacher prompt
    teacher_prompt = f"{prompt}\n正确解:\n{solution}\n再试一次"

    # 3. Tokenize
    teacher_input_ids = tokenizer(teacher_messages, ...)

    return DataProto(teacher_input_ids, teacher_attention_mask, ...)
```

---

## 修复的问题

在开发过程中发现并修复了以下导入路径问题：

| 错误路径 | 正确路径 |
|---------|---------|
| `verl.utils.tensordict_utils.compute_position_id_with_mask` | `verl.utils.model.compute_position_id_with_mask` |
| `verl.trainer.ppo.ray_trainer.Role` | `verl.trainer.ppo.utils.Role` |

---

## 后续工作

1. **测试运行**: 在有 torch/ray 环境的机器上测试导入和运行
2. **Worker 集成**: 当前使用 verl 内置的 Worker，如需自定义可进一步扩展
3. **配置调优**: 根据具体任务调整参数

---

## 参考资料

- verl 官方仓库: https://github.com/volcengine/verl
- SDPO 论文: arXiv:2601.20802 "Reinforcement Learning via Self-Distillation"
- verl recipe 示例: `recipe/spin/`, `recipe/dapo/`
