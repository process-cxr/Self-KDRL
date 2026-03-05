# SDPO 训练与验证指标详解

## 目录
- [一、验证指标 (Validation Metrics)](#一验证指标-validation-metrics)
- [二、训练指标 (Training Metrics)](#二训练指标-training-metrics)
- [三、性能指标 (Performance Metrics)](#三性能指标-performance-metrics)
- [四、SDPO 特有指标](#四sdpo-特有指标)
- [五、训练观察重点](#五训练观察重点)

---

# 一、验证指标 (Validation Metrics)

## 1.1 验证流程概览

每次验证时（按 `test_freq` 配置），会对固定验证集中的每个问题生成多个回答：

```
验证集（固定的问题，来自 test.parquet）：
    问题1: "1+1等于几？"
    问题2: "2+2等于几？"
    问题3: "3+3等于几？"
    ...

验证配置：val_kwargs.n=4

对每个问题生成 4 个不同的回答：
    问题1 → [回答1_1, 回答1_2, 回答1_3, 回答1_4]  → 4个分数
    问题2 → [回答2_1, 回答2_2, 回答2_3, 回答2_4]  → 4个分数
    ...
```

## 1.2 分组和统计

每个问题（uid）有 4 个回答，先对单个问题计算统计量，再对所有问题取平均：

```python
# 单个问题的 4 个回答示例
reward = [0.5, 1.0, 0.0, 1.0]  # 4个回答的分数
pred   = ["A", "A", "B", "A"]   # 4个回答的预测

# 计算统计量
mean@4  = 0.625      # (0.5+1.0+0.0+1.0)/4
best@4  ≈ 0.975      # Bootstrap 采样4个取最大值的期望
maj@4   = 1.0        # pred="A"出现3次，取A的分数

# 对所有问题取平均
val-core/tooluse/reward/mean@4 = mean([问题1.mean, 问题2.mean, ...])
```

## 1.3 统计量含义

### mean（平均值）

| 指标 | 含义 |
|------|------|
| `val-core/tooluse/reward/mean@4` | 4个回答的平均奖励（最常用） |
| `val-aux/tooluse/reward/mean@2` | 前2个回答的平均 |

**含义：** 模型的期望性能，代表"一次尝试"的平均表现

---

### best（最佳值）

| 指标 | 含义 |
|------|------|
| `val-core/tooluse/reward/best@4/mean` | 4个回答中最佳的期望 |
| `val-core/tooluse/reward/best@4/std` | 最佳值的标准差 |
| `val-aux/tooluse/reward/best@2/mean` | 前2个的最佳期望 |

**含义：** 模型能力的上限潜力，模拟"多次尝试选最优"的场景

---

### worst（最差值）

| 指标 | 含义 |
|------|------|
| `val-aux/tooluse/reward/worst@4/mean` | 4个中最差的期望 |
| `val-aux/tooluse/reward/worst@4/std` | 最差值的标准差 |

**含义：** 模型的下限，评估最差情况下的鲁棒性

---

### maj（多数投票）

| 指标 | 含义 |
|------|------|
| `val-core/tooluse/reward/maj@4/mean` | 4个回答投票后的期望 |
| `val-core/tooluse/reward/maj@4/std` | 投票结果的标准差 |

**含义：** 模拟"集成多个预测，投票选择"的场景，需要 reward function 返回 pred 字段

---

### std（标准差）

| 指标 | 含义 |
|------|------|
| `val-aux/tooluse/reward/std@2` | 前2个回答的标准差 |
| `val-aux/tooluse/reward/std@4` | 4个回答的标准差 |

**含义：** 输出的稳定性，越小越稳定

---

## 1.4 指标分类

### val-core（核心指标）

**特点：** 模型性能的主要关注点，用于 WandB 主图展示

```
val-core/tooluse/reward/mean@4          ← 最常用
val-core/tooluse/reward/best@4/mean
val-core/tooluse/reward/best@4/std
val-core/tooluse/reward/maj@4/mean
val-core/tooluse/reward/maj@4/std
```

### val-aux（辅助指标）

**特点：** 帮助诊断的补充指标

```
# @2 指标
val-aux/tooluse/reward/mean@2
val-aux/tooluse/reward/std@2
val-aux/tooluse/reward/best@2/mean
val-aux/tooluse/reward/best@2/std
val-aux/tooluse/reward/maj@2/mean
val-aux/tooluse/reward/maj@2/std

# @4 其他指标
val-aux/tooluse/reward/std@4
val-aux/tooluse/reward/worst@4/mean
val-aux/tooluse/reward/worst@4/std

# 其他
val-aux/num_turns/min/max/mean
```

## 1.5 采样大小对比 (@2 vs @4)

**@2 不是只生成 2 个回答，而是用 4 个回答中的前 2 个来计算！**

| 对比 | 差距大 → 含义 | 差距小 → 含义 |
|------|--------------|--------------|
| `mean@2 vs mean@4` | 输出不稳定，后2个质量低 | 输出稳定，前后一致 |
| `best@2 vs best@4` | 随机性高，需要更多采样 | 随机性低，少量采样即可 |

---

# 二、训练指标 (Training Metrics)

训练指标在每步训练时记录，反映模型在训练过程中的状态。

## 2.1 GRPO 奖励指标（非 Critic）

**重要：** SDPO 使用 GRPO，不需要传统的 Critic 网络。以下指标由 GRPO 计算。

### 数据流

```
reward_function → token_level_scores (原始奖励)
                    ↓
          use_kl_in_reward: false (SDPO)
                    ↓
          token_level_rewards = token_level_scores
                    ↓
                  GRPO 计算
                    ↓
         advantages = score - group_mean (相对优势)
         returns = advantages (直接相等)
```

### critic/score（序列分数）

| 指标 | 含义 |
|------|------|
| `critic/score/mean` | 序列分数的平均值 |

**说明：** `score.sum()`，反映回答整体质量。

---

### critic/rewards（序列奖励）

| 指标 | 含义 |
|------|------|
| `critic/rewards/mean` | 序列奖励的平均值 |

**说明：** SDPO 中 `rewards = scores`（因为 `use_kl_in_reward: false`）。

---

### critic/advantages（优势函数）

| 指标 | 含义 |
|------|------|
| `critic/advantages/mean` | 优势函数的平均值 |
| `critic/advantages/max` | 优势函数的最大值 |
| `critic/advantages/min` | 优势函数的最小值 |

**计算方式（GRPO）：**
```python
# 同一问题（uid）的 n=8 个回答
advantage_i = score_i - mean(scores_in_group)
```

**说明：**
- 反映"我比别人好多少"
- 理论上 `mean ≈ 0`（正负抵消）
- 非零说明分组有问题

---

### critic/returns（回报）

| 指标 | 含义 |
|------|------|
| `critic/returns/mean` | 回报的平均值 |

**说明：** GRPO 中 `returns = advantages`，没有传统的回报计算。

---

### 不存在的指标

| 指标 | 状态 | 原因 |
|------|------|------|
| `critic/values/*` | ❌ 不存在 | SDPO 不使用 Critic 网络 |
| `critic/vf_explained_var` | ❌ 不存在 | SDPO 不使用 Critic 网络 |

---

### 示例

同一问题的 4 个回答：
```
回答分数: [1.0, 0.0, 1.0, 0.0]
group_mean = 0.5

advantages = [0.5, -0.5, 0.5, -0.5]
returns = [0.5, -0.5, 0.5, -0.5]

advantages/mean = 0  ← 应该接近 0
```

---

## 2.2 数据相关指标

### response_length（响应长度）

| 指标 | 含义 |
|------|------|
| `response_length/mean` | 响应 token 数的平均值 |
| `response_length/max` | 响应 token 数的最大值 |
| `response_length/min` | 响应 token 数的最小值 |
| `response_length/clip_ratio` | 触发最大长度限制的比例 |

**说明：** 反映模型输出长度分布，clip_ratio 高说明很多回答被截断。

---

### response_length_non_aborted（非中断响应长度）

| 指标 | 含义 |
|------|------|
| `response_length_non_aborted/mean` | 非中断响应的平均长度 |
| `response_length_non_aborted/max` | 非中断响应的最大长度 |
| `response_length_non_aborted/min` | 非中断响应的最小长度 |
| `response_length_non_aborted/clip_ratio` | 触发截断的比例 |

**说明：** 排除中断（长度=0）的样本，反映有效回答的长度分布。

---

### response/aborted_ratio（中断比例）

| 指标 | 含义 |
|------|------|
| `response/aborted_ratio` | 中断样本的比例 |

**说明：** 中断 = 响应长度 = 0，比例过高说明生成有问题。

---

### prompt_length（提示长度）

| 指标 | 含义 |
|------|------|
| `prompt_length/mean` | 提示 token 数的平均值 |
| `prompt_length/max` | 提示 token 数的最大值 |
| `prompt_length/min` | 提示 token 数的最小值 |
| `prompt_length/clip_ratio` | 触发最大长度限制的比例 |

**说明：** 输入 prompt 的长度分布。

---

### num_turns（对话轮数）

| 指标 | 含义 |
|------|------|
| `num_turns/mean` | 对话轮数的平均值 |
| `num_turns/max` | 对话轮数的最大值 |
| `num_turns/min` | 对话轮数的最小值 |

**说明：** 多轮对话的轮数统计，仅当数据支持多轮时有效。

---

### tool_call_counts（工具调用次数）

| 指标 | 含义 |
|------|------|
| `tool_call_counts/mean` | 工具调用次数的平均值 |
| `tool_call_counts/max` | 工具调用次数的最大值 |
| `tool_call_counts/min` | 工具调用次数的最小值 |

**说明：** 每次对话中工具调用的次数统计。

---

## 2.3 Actor 指标

### actor/entropy（熵）

| 指标 | 含义 |
|------|------|
| `actor/entropy` | 输出分布的平均熵 |

**说明：**
- 熵越大 = 输出越多样化
- 熵越小 = 输出越确定
- 训练初期熵较大，后期熵下降

---

### actor/policy_loss（策略损失）

| 指标 | 含义 |
|------|------|
| `actor/policy_loss` | PPO 策略损失 |

**说明：** 策略网络的损失，反映训练梯度的大小。

---

### actor/kl（KL 散度）

| 指标 | 含义 |
|------|------|
| `actor/kl` | 当前策略 vs 初始策略的 KL 散度 |

**说明：**
- 反映策略变化的幅度
- KL 太大 = 策略变化过快，可能崩溃
- KL 太小 = 策略变化太慢，训练效率低
- 通常控制在 0.01-0.1 范围

---

### actor/clip_frac（截断比例）

| 指标 | 含义 |
|------|------|
| `actor/clip_frac` | 触发 PPO clip 的比例 |

**说明：**
- PPO clip 防止策略更新过大
- clip_frac ≈ 0.5 表示更新正常
- clip_frac 接近 0 或 1 可能需要调整 clip range

---

### actor/reward_kl_penalty（奖励 KL 惩罚）

| 指标 | 含义 |
|------|------|
| `actor/reward_kl_penalty` | 奖励中的 KL 惩罚值 |
| `actor/reward_kl_penalty_coeff` | KL 惩罚系数 |

**说明：** 当 `use_kl_in_reward=True` 时使用，控制策略偏离初始策略的程度。

---

## 2.4 Rollout Correction 指标

Rollout Correction 用于纠正 rollout 和 update 之间的分布偏差。

### rollout_is（Importance Sampling 权重）

| 指标 | 含义 |
|------|------|
| `rollout_is/mean` | IS 权重的平均值 |
| `rollout_is/std` | IS 权重的标准差 |
| `rollout_is/max` | IS 权重的最大值 |
| `rollout_is/min` | IS 权重的最小值 |
| `rollout_is/eff_sample_size` | 有效样本大小 |
| `rollout_is/seq_mean` | 序列 IS 权重的平均值 |
| `rollout_is/seq_std` | 序列 IS 权重的标准差 |
| `rollout_is/max_deviation` | 最大偏差 |

**说明：**
- IS 权重 = p_update / p_rollout，纠正 rollout 和 update 的分布差异
- mean ≈ 1 表示分布差异不大
- std 太大说明分布差异大，训练不稳定
- eff_sample_size 越小说明偏差越大

---

### rollout_rs（Rollout Sampling 权重）

| 指标 | 含义 |
|------|------|
| `rollout_rs/mean` | RS 权重的平均值 |
| `rollout_rs/std` | RS 权重的标准差 |
| `rollout_rs/max` | RS 权重的最大值 |
| `rollout_rs/min` | RS 权重的最小值 |
| `rollout_rs/eff_sample_size` | 有效样本大小 |
| `rollout_rs/seq_mean` | 序列 RS 权重的平均值 |
| `rollout_rs/seq_std` | 序列 RS 权重的标准差 |

**说明：** 类似 IS 权重，用于 Rollout Sampling 策略。

---

### Perplexity（困惑度）

| 指标 | 含义 |
|------|------|
| `training_ppl` | 训练时的困惑度 |
| `rollout_ppl` | rollout 时的困惑度 |
| `ppl_ratio` | rollout_ppl / training_ppl |
| `log_ppl_diff` | log(rollout_ppl) - log(training_ppl) |
| `chi2_token` | token 级别的卡方统计 |
| `chi2_seq` | 序列级别的卡方统计 |

**说明：**
- ppl 越低 = 模型越确定，输出质量越好
- ppl_ratio ≈ 1 表示 rollout 和 update 分布一致
- ppl_ratio > 1 表示 rollout 分布退化

---

### kl（KL 散度）

| 指标 | 含义 |
|------|------|
| `kl` | KL(rollout || update) |
| `k3_kl` | K3 近似的 KL 散度 |

**说明：** 反映 rollout 和 update 策略的分布差异。

---

# 三、性能指标 (Performance Metrics)

## 3.1 Timing（时间）

### timing_s（秒）

| 指标 | 含义 |
|------|------|
| `timing_s/gen` | 生成阶段时间 |
| `timing_s/ref` | 参考模型计算时间 |
| `timing_s/values` | Critic 计算时间 |
| `timing_s/adv` | 优势计算时间 |
| `timing_s/update_critic` | Critic 更新时间 |
| `timing_s/update_actor` | Actor 更新时间 |
| `timing_s/step` | 总步时间 |

---

### timing_per_token_ms（毫秒/token）

| 指标 | 含义 |
|------|------|
| `timing_per_token_ms/gen` | 生成每 token 时间 |
| `timing_per_token_ms/ref` | 参考模型每 token 时间 |
| `timing_per_token_ms/values` | Critic 每 token 时间 |
| `timing_per_token_ms/adv` | 优势计算每 token 时间 |
| `timing_per_token_ms/update_critic` | Critic 更新每 token 时间 |
| `timing_per_token_ms/update_actor` | Actor 更新每 token 时间 |

**说明：** 越小表示效率越高，用于性能分析和瓶颈定位。

---

## 3.2 Throughput（吞吐量）

| 指标 | 含义 |
|------|------|
| `perf/total_num_tokens` | 处理的总 token 数 |
| `perf/time_per_step` | 每步时间 |
| `perf/throughput` | tokens/s/GPU |

**说明：** throughput 是最重要的效率指标，表示每 GPU 每秒处理的 token 数。

---

# 四、SDPO 特有指标

Self-Distillation 相关指标，反映自蒸馏过程的效果。

## 4.1 self_distillation 指标

| 指标 | 含义 |
|------|------|
| `self_distillation/success_group_fraction` | 有成功轨迹的问题比例 |
| `self_distillation/success_sample_fraction` | 使用成功轨迹的样本比例 |
| `self_distillation/feedback_available_fraction` | 有反馈可用的样本比例 |
| `self_distillation/feedback_used_fraction` | 实际使用反馈的样本比例 |
| `self_distillation/reprompt_sample_fraction` | 使用 reprompt 的样本比例 |

---

### success_group_fraction（成功组比例）

**含义：** 同一个问题（uid）下，至少有一个样本奖励 ≥ success_threshold 的比例

**解读：**
- ≈ 1：大多数问题都有成功轨迹，自蒸馏信号丰富
- ≈ 0：很少有成功轨迹，自蒸馏信号稀缺

---

### success_sample_fraction（成功样本比例）

**含义：** 所有样本中，成功轨迹（奖励 ≥ threshold）的比例

**解读：**
- 比 success_group_fraction 更细粒度
- 反映单个 batch 中的成功样本密度

---

### feedback_available_fraction（反馈可用比例）

**含义：** reward function 返回反馈的样本比例

**解读：**
- 取决于数据集和 reward function
- tooluse 数据集可能有反馈

---

### feedback_used_fraction（反馈使用比例）

**含义：** 实际在 reprompt 中使用反馈的样本比例

**解读：**
- 可能 < feedback_available_fraction
- 受 `include_environment_feedback` 和 `environment_feedback_only_without_solution` 配置影响

---

### reprompt_sample_fraction（Reprompt 样本比例）

**含义：** 使用 reprompt（有成功轨迹或使用反馈）的样本比例

**解读：**
- = success_sample_fraction + 反馈使用比例（考虑配置）
- 越高说明自蒸馏信号越多

---

# 五、训练观察重点

## 5.1 核心关注指标

| 类别 | 指标 | 期望趋势 | 含义 |
|------|------|---------|------|
| 验证 | `val-core/tooluse/reward/mean@4` | ↑ 持续上升 | 模型能力提升 |
| 验证 | `val-aux/tooluse/reward/std@4` | ↓ 持续下降 | 输出越来越稳定 |
| 训练 | `critic/score/mean` | ↑ 上升 | 训练奖励增加 |
| 训练 | `critic/advantages/mean` | ≈ 0 | GRPO 正常 |
| 训练 | `actor/entropy` | → 适度下降 | 输出趋于确定 |
| 训练 | `actor/kl` | → 稳定在 0.01-0.1 | 策略变化适中 |
| 效率 | `rollout_is/mean` | ≈ 1 | Rollout 无退化 |
| 效率 | `perf/throughput` | ↑ 越高越好 | 训练效率 |

## 5.2 训练阶段诊断

### 早期阶段（前 10% epochs）
```
val-core/reward/mean@4:     0.2 → 0.4  (快速上升)
critic/rewards/mean:        0.1 → 0.3  (快速上升)
actor/kl:                  0.02 → 0.05 (逐渐增大)
actor/entropy:              2.0 → 1.5  (适度下降)
self_distillation/success_group_fraction: 0.3 → 0.5 (信号增加)
```
**诊断：** 模型快速学习，熵下降，策略开始变化，自蒸馏信号逐渐丰富。

---

### 中期阶段（10%-60% epochs）
```
val-core/reward/mean@4:     0.4 → 0.6  (稳步上升)
critic/rewards/mean:        0.3 → 0.5  (稳步上升)
actor/kl:                  0.05 (稳定)
critic/advantages/mean:     ≈ 0 (保持稳定)
val-aux/reward/std@4:       0.4 → 0.2  (输出稳定)
self_distillation/success_group_fraction: 0.6 → 0.8
```
**诊断：** 训练进展良好，输出趋于稳定。

---

### 后期阶段（60%-100% epochs）
```
val-core/reward/mean@4:     0.6 → 0.65 (缓慢上升)
critic/rewards/mean:        0.5 → 0.52 (接近收敛)
actor/kl:                  0.03 → 0.02 (稳定下降)
critic/advantages/mean:     ≈ 0 (保持稳定)
val-aux/reward/std@4:       0.2 → 0.15 (非常稳定)
```
**诊断：** 接近收敛，考虑调整学习率或停止训练。

---

## 5.3 异常情况诊断

| 异常现象 | 可能原因 | 检查建议 |
|---------|---------|---------|
| val reward 持续下降 | 学习率过大/数据问题 | 检查 `critic/rewards/mean` |
| val reward 波动大 | 数据不稳定/batch 太小 | 增大 `train_batch_size` |
| actor/kl 持续增大 | 学习率过大 | 降低 `actor.optim.lr` |
| actor/kl 接近 0 | 学习率过小 | 提高 `actor.optim.lr` |
| critic/advantages/mean 远离 0 | 分组有问题 | 检查 uid 分组 |
| response/aborted_ratio > 0.5 | 生成有问题 | 检查生成配置 |
| rollout_is/std 很大 | rollout/update 分布差异大 | 降低 temperature |
| self_distillation/success_group_fraction < 0.1 | 成功轨迹太少 | 检查数据质量 |

---

## 5.4 快速参考

| 问题 | 查看指标 |
|------|---------|
| 模型整体性能？ | `val-core/tooluse/reward/mean@4` |
| 模型能力上限？ | `val-core/tooluse/reward/best@4/mean` |
| 输出是否稳定？ | `val-aux/tooluse/reward/std@4` |
| 训练是否有效？ | `critic/score/mean` |
| 策略是否稳定？ | `actor/kl` |
| GRPO 是否正常？ | `critic/advantages/mean` (应≈0) |
| Rollout 是否退化？ | `rollout_is/mean` (应≈1) |
| 自蒸馏信号？ | `self_distillation/success_group_fraction` |
| 训练效率？ | `perf/throughput` |

---

## 5.5 配置参数影响

| 参数 | 影响指标 | 建议 |
|------|---------|------|
| `lr` | actor/kl, val reward | 1e-5 到 5e-5 |
| `temperature` | rollout_is/std, val std | 0.6-1.0 |
| `clip_range` | actor/clip_frac | 0.1-0.2 |
| `entropy_coeff` | actor/entropy | 0.0-0.01 |
| `val_kwargs.n` | 验证指标精度 | 4-8 |
| `train_batch_size` | 训练稳定性 | 32-128 |
