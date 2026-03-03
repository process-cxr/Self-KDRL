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

# 3. 计算 JSD 损失（或其他 KL 变体）
M = (1-α) * P_student + α * P_teacher
JSD = (1-α) * KL(M || P_student) + α * KL(M || P_teacher)

L_SDPO = JSD
```

**特点**：
- 损失是 **Token-level**，每个位置有不同的学习目标
- 教师分布来自成功轨迹，提供更精细的信用分配
- JSD 比 KL 更稳定，避免模式坍塌
- 支持 Forward KL、Reverse KL、JSD 三种变体

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
| **损失函数** | $L = -\min(r \cdot A, \text{clip}(r) \cdot A)$ | $L = \text{JSD}(P_s \| P_t)$ 或 KL 变体 |
| **KL 散度** | 可选（作为正则项） | 核心（蒸馏损失） |
| **Teacher** | 无 | $P_t$ 来自成功轨迹 |
| **学习信号粒度** | 序列级别 | Token 级别 |

---

## 二、代码层面对比

### 2.1 文件改动对比

| 文件 | GRPO 改动 | SDPO 改动 |
|------|----------|----------|
| `core_algos.py` | 新增 `compute_grpo_outcome_advantage()` | 新增 `compute_self_distillation_loss()` |
| `dp_actor.py` | 无改动 | 新增 `SDPODataParallelPPOActor`, `TrustRegionTeacher` |
| `fsdp_workers.py` | 无改动 | 新增 `SDPOActorRolloutRefWorker`, `AsyncSDPOActorRolloutRefWorker` |
| `ray_trainer.py` | 无改动 | 新增 `RaySDPOTrainer` (继承扩展) |
| `config/` | 新增 `baseline_grpo.yaml` | 新增 `sdpo_trainer.yaml`, `SelfDistillationConfig` |
| `reward_score/` | 基础奖励函数 | 带反馈的奖励函数 (code, math, tooluse 等) |

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
    2. 计算 JSD 散度（或其他 KL 变体）
    3. 可选：IS clip、Top-k 蒸馏
    """
    alpha = self_distillation_config.alpha  # 0.5 for JSD, 1.0 for reverse KL, 0.0 for forward KL

    if self_distillation_config.full_logit_distillation:
        # Full logit distillation
        if alpha == 0.0:
            kl_loss = F.kl_div(student_distill_log_probs, teacher_distill_log_probs, ...)
        elif alpha == 1.0:
            kl_loss = F.kl_div(teacher_distill_log_probs, student_distill_log_probs, ...)
        else:
            # JSD
            mixture = torch.logsumexp([
                student_distill_log_probs + torch.log(1-alpha),
                teacher_distill_log_probs + torch.log(alpha)
            ], dim=0)
            kl_loss = torch.lerp(
                F.kl_div(mixture, teacher_distill_log_probs, ...),
                F.kl_div(mixture, student_distill_log_probs, ...),
                alpha
            )
    else:
        # Token-only distillation (reverse KL only)
        log_ratio = student_log_probs - teacher_log_probs
        kl_loss = log_ratio.detach() * student_log_probs

    return agg_loss(kl_loss, ...), metrics
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
    grad_norm = self._optimizer_step()
    return metrics

# ==================== SDPO ====================
def update_policy(self, data):
    # 检查是否启用 SDPO
    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
    self_distillation_enabled = loss_mode == "sdpo"

    # 前向传播 (Student)
    outputs = self._forward_micro_batch(model_inputs, ...)
    log_prob = outputs["log_probs"]
    student_all_logps = outputs.get("all_logps")  # SDPO 可能需要完整分布

    if self_distillation_enabled:
        # 前向传播 (Teacher) - SDPO 特有
        teacher_inputs = {
            "responses": model_inputs["responses"],
            "input_ids": model_inputs["teacher_input_ids"],
            "attention_mask": model_inputs["teacher_attention_mask"],
            "position_ids": model_inputs["teacher_position_ids"],
        }
        with torch.no_grad():
            teacher_outputs = self._forward_micro_batch(
                teacher_inputs,
                module=teacher_model,
                return_all_logps=return_all_logps,
                distill_topk=distill_topk,
                ...
            )

        teacher_log_prob = teacher_outputs["log_probs"]
        teacher_all_logps = teacher_outputs.get("all_logps")

        # 计算蒸馏损失 - SDPO 特有
        pg_loss, pg_metrics = compute_self_distillation_loss(
            student_log_probs=log_prob,
            teacher_log_probs=teacher_log_prob,
            response_mask=response_mask,
            self_distillation_config=self_distillation_cfg,
            ...
        )
    else:
        # Vanilla PPO / 其他损失模式
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            ...
        )

    # 反向传播
    loss.backward()
    grad_norm = self._optimizer_step()

    # EMA 更新教师 - SDPO 特有
    if self_distillation_enabled and did_update:
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
│              JSD/KL Distillation Loss                                     │
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
      full_logit_distillation: true
      alpha: 0.5                    # JSD (0=Forward KL, 1=Reverse KL)
      success_reward_threshold: 1.0 # 成功阈值
      teacher_regularization: ema   # EMA 教师
      teacher_update_rate: 0.05     # EMA 率
      distillation_topk: null       # 或 100 (Top-k 蒸馏)
      distillation_add_tail: true
      max_reprompt_len: 10240
      include_environment_feedback: false
      is_clip: null                 # 或 2.0 (IS clip)
  rollout:
    n: 8
```

---

## 三、关键差异总结

### 3.1 算法层面

| 差异点 | GRPO | SDPO |
|--------|------|------|
| **信用分配粒度** | 序列级别 | Token 级别 |
| **教师信号** | 无 | 自身成功轨迹 |
| **损失类型** | PPO-clip | JSD/KL 蒸馏 |
| **优势估计** | 必需 | 可选（蒸馏时不需要） |
| **反馈利用** | 不利用 | 可利用环境反馈 |
| **训练稳定性** | 依赖 clip | JSD 更稳定 |
| **pg_loss 含义** | PPO clipped loss | Student-Teacher KL (蒸馏损失) |
| **kl_loss 含义** | 可选的参考策略 KL | 通常为 0（use_kl_loss=False） |

### 3.2 代码层面

| 差异点 | GRPO | SDPO |
|--------|------|------|
| **核心函数** | `compute_grpo_outcome_advantage()` | `compute_self_distillation_loss()` |
| **Trainer 改动** | 无 | 新增 `RaySDPOTrainer` |
| **Actor 改动** | 无 | 新增 `SDPODataParallelPPOActor` |
| **Worker 改动** | 无 | 新增 `SDPOActorRolloutRefWorker` |
| **前向传播次数** | 1 次 | 2 次 (Student + Teacher) |
| **额外内存** | 无 | 需要存储 Teacher logits |
| **配置项** | 少 | 多 (alpha, topk, is_clip, templates...) |
| **代码隔离** | 在 verl 源码中 | 在 recipe/sdpo/ 目录中 |

### 3.3 计算开销对比

| 开销项 | GRPO | SDPO |
|--------|------|------|
| **前向传播** | O(bs × seq_len × vocab) | O(2 × bs × seq_len × vocab) |
| **反向传播** | 标准 | 标准 + EMA 更新 |
| **内存占用** | 标准 | 需要额外存储 Teacher 输出 |
| **Top-k 优化** | 不支持 | 支持（可减少到 O(k)） |

---

## 四、Metrics 说明

### 4.1 GRPO Metrics

```python
{
    "actor/pg_loss": 0.123,      # PPO clipped loss
    "actor/entropy": 2.456,      # 策略熵
    "actor/kl": 0.012,           # 与参考策略的 KL（如果启用）
    "actor/grad_norm": 1.234,    # 梯度范数
}
```

### 4.2 SDPO Metrics

```python
{
    "actor/pg_loss": 0.445,      # ← 这是 Student-Teacher 的 JSD/KL 蒸馏损失
    "actor/kl_loss": 0.0,        # ← 通常是 0（use_kl_loss=False）
    "actor/entropy": 2.456,
    "actor/grad_norm": 0.968,

    # SDPO 特有指标
    "self_distillation/empty_target_batch": [False, False, True, ...],
    "self_distillation/success_sample_fraction": 0.75,
    "self_distillation/feedback_used_fraction": 0.30,

    # Rollout correction 指标（如果启用）
    "rollout_corr/training_ppl": [...],
    "rollout_corr/rollout_ppl": [...],
}
```

**重要**：在 SDPO 模式下，`pg_loss` 就是蒸馏损失（Student-Teacher KL/JSD），而不是 PPO clipped loss。`kl_loss` 仍然是传统参考策略 KL，默认不启用（值为 0）。

---

## 五、代码实现细节

### 5.1 类继承关系

```
官方 verl:
├── RayPPOTrainer
├── DataParallelPPOActor
├── ActorRolloutRefWorker
└── AsyncActorRolloutRefWorker

recipe/sdpo (扩展):
├── RaySDPOTrainer extends RayPPOTrainer
├── SDPODataParallelPPOActor extends DataParallelPPOActor
├── SDPOActorRolloutRefWorker extends ActorRolloutRefWorker
└── AsyncSDPOActorRolloutRefWorker extends AsyncActorRolloutRefWorker
```

### 5.2 self_distillation_cfg 传递

由于官方 `ActorConfig` 没有 `self_distillation` 字段，recipe 使用参数传递方式：

```python
# fsdp_workers.py
class SDPOActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, config, role, **kwargs):
        # 保存 self_distillation 配置
        self._self_distillation_cfg = config.actor.get("self_distillation", None)

        # 临时从 config 中删除（因为官方 ActorConfig 不支持）
        if "self_distillation" in config.actor:
            del config.actor.self_distillation

        super().__init__(config, role, **kwargs)

    def init_model(self):
        # 通过参数传递给 actor
        self.actor = SDPODataParallelPPOActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
            self_distillation_cfg=self._self_distillation_cfg,  # ← 参数传递
        )
```

### 5.3 Teacher Module 设置

```python
# fsdp_workers.py - init_model 方法
if self._is_actor and self._sdpo_loss_mode == "sdpo":
    teacher_regularization = self._self_distillation_cfg.get("teacher_regularization", "ema")

    if teacher_regularization == "trust-region":
        # 使用 TrustRegionTeacher 包装器
        self.actor.teacher_module = TrustRegionTeacher(
            ref_module=self.ref_module_fsdp,
            student_module=self.actor_module_fsdp,
            mix_coef=self._self_distillation_cfg.get("teacher_update_rate", 0.0),
        )
    else:
        # 直接使用 ref_module 作为教师
        self.actor.teacher_module = self.ref_module_fsdp
```

### 5.4 Teacher EMA 更新

```python
# dp_actor.py
def _update_teacher(self) -> None:
    """EMA 更新教师权重"""
    update_rate = self_distillation_cfg.get("teacher_update_rate", 0.0)
    if update_rate == 0.0:
        return

    with torch.no_grad():
        for teacher_param, student_param in zip(
            self.teacher_module.parameters(),
            self.actor_module.parameters(),
        ):
            student_data = student_param.data.to(device=teacher_param.device)
            teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
```

---

## 六、代码改动清单

### 6.1 SDPO 改动文件（Recipe 隔离模式）

```
recipe/sdpo/
├── __init__.py
│   └── 导出: compute_self_distillation_loss, SDPODataParallelPPOActor, ...
│
├── main_sdpo.py
│   └── SDPO 入口，使用 SDPOWorker 和 RaySDPOTrainer
│
├── config.py
│   └── SelfDistillationConfig 数据类
│   └── validate_sdpo_config 函数
│
├── sdpo_trainer.py
│   └── RaySDPOTrainer extends RayPPOTrainer
│       └── 覆盖: _update_actor()
│       └── 新增: _init_sdpo_config(), _maybe_build_self_distillation_batch()
│       └── 新增: _collect_solutions_by_uid(), _collect_feedback(), _get_solution()
│
├── core_algos.py
│   └── get_policy_loss_fn(name) - 支持 "sdpo" 模式
│   └── compute_self_distillation_loss() - 核心蒸馏损失函数
│
├── dp_actor.py
│   └── TrustRegionTeacher - Trust-region 教师包装器
│   └── SDPODataParallelPPOActor extends DataParallelPPOActor
│       └── 覆盖: update_policy(), _forward_micro_batch(), compute_log_prob()
│       └── 新增: _update_teacher(), _has_non_empty_multi_modal_inputs()
│
├── fsdp_workers.py
│   └── SDPOActorRolloutRefWorker extends ActorRolloutRefWorker (同步)
│       └── 覆盖: init_model(), update_actor(), compute_log_prob()
│   └── AsyncSDPOActorRolloutRefWorker (异步)
│       └── 继承 SDPOActorRolloutRefWorker + AsyncActorRolloutRefWorker
│
├── config/sdpo_trainer.yaml
│   └── SDPO 配置文件
│
└── reward_score/
    ├── __init__.py - 自动分发奖励函数
    ├── code.py - 代码任务奖励 + 反馈
    ├── math.py - 数学任务奖励 + 反馈
    ├── gpqa.py - GPQA 奖励
    ├── mcq.py - 多选题奖励
    ├── mmlu_pro.py - MMLU-Pro 奖励
    └── tooluse.py - ToolUse 奖励 + 反馈
```

### 6.2 依赖官方 verl 的方式

```
recipe/sdpo 通过继承扩展官方 verl，不修改任何源码:

┌─────────────────────────────────────────────────────────────┐
│                   官方 verl (无 SDPO)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ RayPPOTrainer   │  │ DataParallel... │  │ActorRollout  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          ↓ 继承
┌─────────────────────────────────────────────────────────────┐
│                   recipe/sdpo                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ RaySDPOTrainer  │  │ SDPOData...     │  │ SDPOActor... │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 七、何时选择 SDPO vs GRPO

### 选择 GRPO 的场景

- 奖励信号充足且可靠
- 计算资源有限
- 不需要精细的 token-level 信用分配
- 快速迭代实验
- 不关心环境反馈

### 选择 SDPO 的场景

- 需要精细的信用分配
- 有丰富的环境反馈（如代码测试结果）
- 希望利用成功轨迹作为示范
- 追求更高的样本效率
- 有足够的计算资源
- 训练初期成功率较低时更稳定

---

## 八、常见问题

### Q1: SDPO 的 pg_loss 为什么不是 PPO loss?

A: 在 SDPO 模式下，`pg_loss` 是 Student 和 Teacher 之间的蒸馏损失（JSD/KL），不是 PPO clipped loss。这是 SDPO 的核心学习信号。

### Q2: SDPO 的 kl_loss 为什么总是 0?

A: `kl_loss` 是传统参考策略 KL penalty，在 SDPO 模式下默认不启用（`use_kl_loss=False`），所以保持 0。SDPO 的 KL 损失体现在 `pg_loss` 中。

### Q3: SDPO 需要 ref policy 吗?

A: 需要。SDPO 使用 ref policy 作为初始的 teacher（通过 EMA 更新逐步演变）。如果配置了 `trust-region`，teacher 是 ref 和 student 的插值。

### Q4: self_distillation/empty_target_batch 是什么?

A: 显示哪些 micro batch 没有成功样本可用于蒸馏。值为 True 表示该 batch 所有样本的奖励都低于 `success_reward_threshold`，跳过蒸馏。

### Q5: 如何启用环境反馈?

A: 设置 `self_distillation.include_environment_feedback=true`，并确保 reward 函数返回 feedback 字段（如 code.py 中的实现）。
