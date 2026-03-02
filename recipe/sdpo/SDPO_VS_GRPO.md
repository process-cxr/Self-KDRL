# SDPO vs GRPO 算法与代码对比

## 一、算法层面对比

### 1.1 核心思想

| 方面 | GRPO | SDPO |
|------|------|------|
| **全称** | Group Relative Policy Optimization | Self-Distilled Policy Optimization |
| **核心思想** | 用组内均值作为 baseline，无需 Critic | 用自身成功轨迹作为教师信号进行蒸馏 |
| **学习信号** | 标量奖励（序列级别） | Token-level 分布（更精细） |
| **教师来源** | 无 | 自身成功轨迹 + 环境反馈 |

### 1.2 损失函数对比

#### GRPO 损失函数

```
# 1. 计算优势（组内归一化）
A_i = (R_i - mean(R_group)) / std(R_group)

# 2. PPO-clip 损失
L_GRPO = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

# 其中 ratio = π_θ(a|s) / π_old(a|s)
```

**特点**：
- 优势估计是**标量**，整个序列共享一个值
- 依赖 PPO 的 clip 机制防止过大更新
- 无法区分序列中哪些 token 贡献更大

#### SDPO 损失函数

```
# 1. 构建 Teacher 输入
teacher_input = prompt + successful_solution + feedback

# 2. 计算 Student 和 Teacher 的 log probs
log_π_student = Student(original_input)
log_π_teacher = Teacher(teacher_input)

# 3. 计算 JSD 损失
M = (1-α) * P_student + α * P_teacher
JSD = (1-α) * KL(M || P_student) + α * KL(M || P_teacher)

L_SDPO = JSD
```

**特点**：
- 损失是 **Token-level**，每个位置有不同的学习目标
- 教师分布来自成功轨迹，提供更精细的信用分配
- JSD 比 KL 更稳定，避免模式坍塌

### 1.3 信用分配对比

```
GRPO 信用分配（序列级别）:
┌─────────────────────────────────────────────────┐
│  Prompt: "计算 2+2=?"                           │
│                                                 │
│  Response 1: "答案是4"  → R=1.0 → A=+0.5       │
│  Response 2: "答案是5"  → R=0.0 → A=-1.5       │
│  Response 3: "等于4"    → R=1.0 → A=+0.5       │
│                                                 │
│  所有 token 共享同一个优势值 A                   │
└─────────────────────────────────────────────────┘

SDPO 信用分配（Token 级别）:
┌─────────────────────────────────────────────────┐
│  Student Input: "计算 2+2=?"                    │
│  Student Output: "答案是5" (失败)               │
│                                                 │
│  Teacher Input: "计算 2+2=?                     │
│                  正确解: 答案是4                 │
│                  再试一次"                       │
│  Teacher Output: "答案是4" (成功示范)           │
│                                                 │
│  每个 token 都有明确的教师目标分布               │
│  → 更精细的信用分配                              │
└─────────────────────────────────────────────────┘
```

### 1.4 数学公式对比

| 公式 | GRPO | SDPO |
|------|------|------|
| **优势估计** | $A = \frac{R - \mu_G}{\sigma_G}$ | 不需要（直接用蒸馏损失） |
| **损失函数** | $L = -\min(r \cdot A, \text{clip}(r) \cdot A)$ | $L = \text{JSD}(P_s \| P_t)$ |
| **KL 散度** | 可选（作为正则项） | 核心（JSD 蒸馏） |
| **Teacher** | 无 | $P_t$ 来自成功轨迹 |

---

## 二、代码层面对比

### 2.1 文件改动对比

| 文件 | GRPO 改动 | SDPO 改动 |
|------|----------|----------|
| `core_algos.py` | 新增 `compute_grpo_outcome_advantage()` | 新增 `compute_self_distillation_loss()` |
| `dp_actor.py` | 无改动 | 新增 `TrustRegionTeacher`, `_update_teacher()`, 修改 `update_policy()` |
| `ray_trainer.py` | 无改动 | 新增 `_maybe_build_self_distillation_batch()` |
| `config/` | 新增 `baseline_grpo.yaml` | 新增 `sdpo.yaml` |

### 2.2 核心函数对比

#### compute_grpo_outcome_advantage vs compute_self_distillation_loss

```python
# ==================== GRPO ====================
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    """
    GRPO 优势计算：
    1. 计算每个序列的总奖励
    2. 按 UID 分组
    3. 组内归一化
    """
    scores = token_level_rewards.sum(dim=-1)  # [bs]

    # 按 UID 分组计算均值和标准差
    for uid in unique_uids:
        group_scores = scores[index == uid]
        mean = group_scores.mean()
        std = group_scores.std()
        advantages[index == uid] = (scores[index == uid] - mean) / std

    # 扩展到每个 token
    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages

# ==================== SDPO ====================
def compute_self_distillation_loss(
    student_log_probs, teacher_log_probs, response_mask,
    self_distillation_config, ...
):
    """
    SDPO 蒸馏损失：
    1. 计算 Student 和 Teacher 的分布
    2. 计算 JSD 散度
    3. 可选：IS clip、Top-k 蒸馏
    """
    alpha = self_distillation_config.alpha  # 0.5 for JSD

    if alpha == 0.5:
        # JSD
        mixture = torch.logsumexp([
            student_log_probs + log(1-alpha),
            teacher_log_probs + log(alpha)
        ], dim=0)
        kl_teacher = F.kl_div(mixture, teacher_log_probs)
        kl_student = F.kl_div(mixture, student_log_probs)
        loss = torch.lerp(kl_student, kl_teacher, alpha)

    return loss, metrics
```

#### update_policy 对比

```python
# ==================== GRPO (vanilla PPO path) ====================
def update_policy(self, data):
    # 前向传播
    outputs = self._forward_micro_batch(model_inputs, ...)
    log_prob = outputs["log_probs"]

    # 计算 PPO 损失
    policy_loss_fn = get_policy_loss_fn("vanilla")  # 或 "gpg", "clip_cov"
    pg_loss, metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,  # GRPO 优势
        response_mask=response_mask,
        ...
    )

    # 反向传播
    loss.backward()
    self._optimizer_step()
    return metrics

# ==================== SDPO ====================
def update_policy(self, data):
    # 前向传播 (Student)
    outputs = self._forward_micro_batch(model_inputs, ...)
    log_prob = outputs["log_probs"]
    student_all_logps = outputs.get("all_logps")  # SDPO 需要完整分布

    # 前向传播 (Teacher) - SDPO 特有
    teacher_inputs = {
        "input_ids": model_inputs["teacher_input_ids"],
        "attention_mask": model_inputs["teacher_attention_mask"],
        "position_ids": model_inputs["teacher_position_ids"],
    }
    with torch.no_grad():
        teacher_outputs = self._forward_micro_batch(teacher_inputs, module=teacher_model, ...)
    teacher_log_prob = teacher_outputs["log_probs"]
    teacher_all_logps = teacher_outputs.get("all_logps")

    # 计算蒸馏损失 - SDPO 特有
    pg_loss, metrics = compute_self_distillation_loss(
        student_log_probs=log_prob,
        teacher_log_probs=teacher_log_prob,
        student_all_log_probs=student_all_logps,
        teacher_all_log_probs=teacher_all_logps,
        ...
    )

    # 反向传播
    loss.backward()
    self._optimizer_step()

    # EMA 更新教师 - SDPO 特有
    self._update_teacher()
    return metrics
```

### 2.3 数据流对比

```
GRPO 数据流:
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│  Prompt    │ ──▶ │  Rollout   │ ──▶ │   Reward   │ ──▶ │  Advantage │
│  (batch)   │     │  n responses│    │  Computation│    │   (GRPO)   │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
                                                                │
                                                                ▼
                                                         ┌────────────┐
                                                         │ PPO Update │
                                                         └────────────┘

SDPO 数据流:
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────────────┐
│  Prompt    │ ──▶ │  Rollout   │ ──▶ │   Reward   │ ──▶ │ Build Teacher Batch│
│  (batch)   │     │  n responses│    │  + Feedback │    │ (成功解 + 反馈)     │
└────────────┘     └────────────┘     └────────────┘     └────────────────────┘
                                                                │
                         ┌──────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           SDPO Update                                   │
│  ┌─────────────────┐              ┌─────────────────┐                  │
│  │ Student Forward │              │ Teacher Forward │                  │
│  │ (original input)│              │ (teacher input) │                  │
│  └────────┬────────┘              └────────┬────────┘                  │
│           │                                │                           │
│           └────────────┬───────────────────┘                           │
│                        ▼                                               │
│              JSD Distillation Loss                                     │
│                        │                                               │
│                        ▼                                               │
│                  Backward + EMA Update Teacher                         │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.4 配置对比

```yaml
# ==================== GRPO 配置 ====================
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true  # 或 false (Dr.GRPO)

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: vanilla  # 标准 PPO
  rollout:
    n: 8  # 每组采样数

# ==================== SDPO 配置 ====================
algorithm:
  adv_estimator: grpo  # 仍然用 GRPO 计算优势（但损失用蒸馏）

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: sdpo  # 启用 SDPO
    self_distillation:
      alpha: 0.5                    # JSD
      success_reward_threshold: 1.0 # 成功阈值
      teacher_regularization: ema   # EMA 教师
      teacher_update_rate: 0.05     # EMA 率
      distillation_topk: null       # 或 100 (Top-k 蒸馏)
      is_clip: null                 # 或 2.0 (IS clip)
      include_environment_feedback: false
  rollout:
    n: 8
    calculate_log_probs: true  # SDPO 需要
```

---

## 三、关键差异总结

### 3.1 算法层面

| 差异点 | GRPO | SDPO |
|--------|------|------|
| **信用分配粒度** | 序列级别 | Token 级别 |
| **教师信号** | 无 | 自身成功轨迹 |
| **损失类型** | PPO-clip | JSD 蒸馏 |
| **优势估计** | 必需 | 可选（蒸馏时不需要） |
| **反馈利用** | 不利用 | 可利用环境反馈 |
| **训练稳定性** | 依赖 clip | JSD 更稳定 |

### 3.2 代码层面

| 差异点 | GRPO | SDPO |
|--------|------|------|
| **新增函数** | `compute_grpo_outcome_advantage()` | `compute_self_distillation_loss()` |
| **Trainer 改动** | 无 | 新增 `_maybe_build_self_distillation_batch()` |
| **Actor 改动** | 无 | 新增 `_update_teacher()`, 修改 `update_policy()` |
| **前向传播次数** | 1 次 | 2 次 (Student + Teacher) |
| **额外内存** | 无 | 需要存储 Teacher logits |
| **配置项** | 少 | 多 (alpha, topk, is_clip, templates...) |

### 3.3 计算开销对比

| 开销项 | GRPO | SDPO |
|--------|------|------|
| **前向传播** | O(bs × seq_len × vocab) | O(2 × bs × seq_len × vocab) |
| **反向传播** | 标准 | 标准 + EMA 更新 |
| **内存占用** | 标准 | 需要额外存储 Teacher 输出 |
| **Top-k 优化** | 不支持 | 支持（可减少到 O(k)） |

---

## 四、优势估计器说明（OTB 不是 SDPO 专属）

### 4.1 优势估计器分类

OTB (Optimal Token Baseline) 是一个**独立的优势估计方法**，与 GRPO、GAE 等是并列关系，**不是 SDPO 的专属改动**：

```
优势估计器 (adv_estimator)
├── GAE          # Generalized Advantage Estimation (需要 Critic)
├── GRPO         # Group Relative Policy Optimization
├── REINFORCE    # 基础 REINFORCE
├── RLOO         # REINFORCE Leave-One-Out
├── OTB          # Optimal Token Baseline ← 独立方法，GRPO/SDPO 都可用
└── ...
```

### 4.2 OTB 可以配合 GRPO 使用

```yaml
# GRPO + OTB 配置示例
algorithm:
  adv_estimator: optimal_token_baseline  # 使用 OTB 而非 GRPO 的优势计算

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: vanilla  # 标准 PPO 损失
```

### 4.3 OTB 的原理

OTB 为每个 timestep 计算一个最优 baseline：

```
B_t* = E[G_t × W_t] / E[W_t]

其中 W_t = Σ_{j=1}^t (1 - 2π_j + Σπ²)  # 累积路径方差代理
```

**优势**：
- Token-level baseline，比 GRPO 的序列级 baseline 更精细
- 理论上最小化优势估计的方差
- 可以和任何策略损失函数组合使用

### 4.4 SDPO 的优势估计

SDPO 默认使用 GRPO 计算优势，但优势值实际上在蒸馏损失中不起主要作用：

```yaml
# SDPO 默认配置
algorithm:
  adv_estimator: grpo  # 虽然计算了优势，但主要损失来自蒸馏

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: sdpo  # 蒸馏损失，不依赖 advantages
```

---

## 五、何时选择 SDPO vs GRPO

### 选择 GRPO 的场景

- 奖励信号充足且可靠
- 计算资源有限
- 不需要精细的 token-level 信用分配
- 快速迭代实验

### 选择 SDPO 的场景

- 需要精细的信用分配
- 有丰富的环境反馈（如代码测试结果）
- 希望利用成功轨迹作为示范
- 追求更高的样本效率
- 有足够的计算资源

---

---

## 六、代码改动清单

### 6.1 GRPO 改动文件（官方 verl）

```
verl/trainer/ppo/core_algos.py
  └── 新增: compute_grpo_outcome_advantage()

verl/trainer/config/baseline_grpo.yaml
  └── 新增: GRPO 配置文件
```

### 6.2 SDPO 改动文件（recipe 隔离模式）

**Recipe 模式（当前实现）** - 完全独立于 verl 源码：

```
recipe/sdpo/
├── dp_actor.py
│   ├── 新增: class TrustRegionTeacher
│   ├── 新增: class SDPODataParallelPPOActor
│   │   └── 覆盖: update_policy(), _forward_micro_batch()
│   │   └── 新增: _update_teacher(), set_teacher_module()
│
├── fsdp_workers.py
│   ├── 新增: class SDPOActorRolloutRefWorker (同步)
│   └── 新增: class AsyncSDPOActorRolloutRefWorker (异步)
│
├── sdpo_trainer.py
│   ├── 新增: class RaySDPOTrainer
│   │   └── 覆盖: _maybe_build_self_distillation_batch()
│   │   └── 新增: _collect_solutions_by_uid()
│
├── core_algos.py
│   └── 新增: compute_self_distillation_loss()
│
├── main_sdpo.py
│   └── 新增: SDPO 配置检查逻辑
│
├── config/sdpo_trainer.yaml
│   └── 新增: SDPO 配置文件
│
└── reward_score/
    ├── code.py (代码奖励 + 反馈)
    └── tooluse.py (ToolUse 奖励 + 反馈)
```

**关键区别**：
- Recipe 模式：所有 SDPO 代码在 `recipe/sdpo/` 目录，不修改 verl 源码
- 原始 verl-sdpo 模式：SDPO 代码分散在 `verl/` 目录各处

### 6.3 依赖官方 verl 的方式

```
recipe/sdpo 通过继承扩展官方 verl:

┌─────────────────────────────────────────────────────────────┐
│                   官方 verl (无 SDPO)                        │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ RayPPOTrainer   │  │ ActorRolloutRef │                 │
│  └─────────────────┘  └─────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
                          ↓ 继承
┌─────────────────────────────────────────────────────────────┐
│                   recipe/sdpo                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │ RaySDPOTrainer  │  │ SDPOActorRollout│  │TrustRegion   │  │
│  └─────────────────┘  │ RefWorker        │  │Teacher        │  │
│                          └─────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---
