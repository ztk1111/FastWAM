# FastWAMIDM Subgoal Latent 第一版改造计划（目标目录：`/data/ztk/code/iclr`）
保持侵入最小化原则，减少防御性编程，有问题就正常报错然后针对性解决
## Summary

将当前 FastWAMIDM 的 Stage 1 从“生成与 action chunk 等长的未来观测/video latent chunk”改为“生成少量 task-relevant subgoal latent”。Action horizon 保持不变，例如 `action_horizon=15`，但 Stage 1 只生成 `num_subgoal_latents=5` 个观测 latent。Stage 2 仍通过 video expert/MoT 的 video-side KV cache 条件化 action denoising；区别是 cache 来自 K 个 subgoal latent 对应的 video tokens，而不是完整 H 帧 latent tokens。

第一版采用固定时间采样构造 subgoal supervision，不做 learned selector，不在推理时完整生成再挑选。目标是以最小工程风险验证：更短、更任务相关的 goal-conditioned latent bottleneck 是否能提升 action chunk 生成稳定性和任务完成率。

## Design Goals

- 输入接口保持接近当前 IDM：当前观测、语言条件、可选本体状态、可选 goal token。
- 输出接口保持 action chunk 不变：`action` 仍为 `[H, action_dim]`。
- Stage 1 输出从完整未来 latent chunk 改成 K 个 subgoal latent：`K <= H`，由配置指定。
- Goal token 主要约束 Stage 1 subgoal latent 生成；Stage 2 通过 subgoal latent 的 KV cache 间接受益。
- 第一版优先稳定和可对比，避免同时引入 learned selector、复杂 loss 或新模型结构导致归因困难。

## Core Behavior

原 IDM：

```text
obs_t + language + proprio + goal_token
  -> Stage 1 denoise video latent chunk length H
  -> video_expert.pre_dit(full video latent)
  -> MoT video KV cache
  -> Stage 2 denoise action chunk length H
```

Subgoal IDM 第一版：

```text
obs_t + language + proprio + goal_token
  -> Stage 1 denoise subgoal latent length K
  -> video_expert.pre_dit(subgoal latent)
  -> MoT subgoal/video KV cache
  -> Stage 2 denoise action chunk length H
```

注意：action decoder 不直接读取 raw latent；它读取的是 subgoal latent 经 video expert/MoT 预填充后的 video-side KV cache。

## Public Config Additions

在 IDM model config 中新增可选块，默认关闭以保持旧行为：

```yaml
subgoal_latent:
  enabled: true
  num_subgoal_latents: 5
  selection_strategy: uniform_endpoints   # uniform_endpoints | uniform_inside | custom
  custom_indices: null                    # 例如 [2, 5, 8, 11, 14]，0-based latent time index
  include_first_latent: false             # 默认首帧/current obs 不作为未来 subgoal target
  action_time_conditioning: true          # 给 action tokens 增加 subgoal-relative time hint
  subgoal_token_dropout: 0.0              # 训练时随机丢 subgoal tokens/cache，增强鲁棒性
  goal_token_dropout: 0.0                 # 训练 Stage 1 时随机 drop goal token，避免过拟合
  teacher_forcing_ratio: 1.0              # 第一版默认 action 使用 GT subgoal cond，可后续退火
```

建议默认实验值：

```yaml
subgoal_latent:
  enabled: true
  num_subgoal_latents: 5
  selection_strategy: uniform_endpoints
  action_time_conditioning: true
  subgoal_token_dropout: 0.05
  goal_token_dropout: 0.10
  teacher_forcing_ratio: 1.0
```

## Key Implementation Changes

### 1. Subgoal index 生成工具

新增一个小工具函数，给定 latent/video chunk 长度 `T` 和 `K`，返回监督用的时间索引：

```python
select_subgoal_indices(
    latent_len: int,
    num_subgoal_latents: int,
    strategy: str,
    custom_indices: list[int] | None = None,
    include_first_latent: bool = False,
) -> torch.Tensor
```

第一版策略：

- `uniform_endpoints`：从未来范围均匀取 K 个点，包含最后一个 latent。
- `uniform_inside`：从未来范围中间均匀取 K 个点，避免过度偏向终态。
- `custom`：使用配置给定的 0-based latent time indices。

当 latent 时间长度来自 VAE 下采样后，不要直接用 action horizon 的帧数索引。应基于 `input_latents.shape[2]` 选择 latent-time 索引。

### 2. 训练目标从 full chunk 改为 subgoal chunk

在 `FastWAMIDM.training_loss()` 中：

- 保持 dataset / processor 输出完整视频 chunk，不第一版修改数据管线。
- `build_inputs()` 后得到 `input_latents: [B, C, T, H, W]`。
- 若 `subgoal_latent.enabled=true`：
  - 生成 `subgoal_indices`。
  - `target_subgoal_latents = input_latents.index_select(dim=2, index=subgoal_indices)`。
  - Stage 1 的 noisy latent / target latent 都基于 `target_subgoal_latents`。
  - `first_frame_latents` 不再强行覆盖 `latents_noisy[:, :, 0:1]`，除非显式 `include_first_latent=true`。
- 若 `subgoal_latent.enabled=false`：保持原 IDM 路径。

关键点：第一版训练时 action branch 的 video condition 建议使用 GT subgoal latent，而不是 Stage 1 predicted latent，以降低训练不稳定和 diffusion train-test 混杂。后续再做 scheduled sampling。

### 3. 推理 Stage 1 只初始化和去噪 K 个 latent

在 `FastWAMIDM.infer_joint()` 中：

- 原本 `latent_t` 由 `num_video_frames` 决定。
- 若 `subgoal_latent.enabled=true`，Stage 1 denoise 的 latent time length 改为 `K`：

```python
latents_subgoal = torch.randn((1, C, K, latent_h, latent_w), ...)
```

- 不再把首帧 latent 写入 `latents_subgoal[:, :, 0:1]`，除非 `include_first_latent=true`。
- Stage 1 denoise 调用仍传 `extra_context_emb=video_goal_hidden`。
- Stage 2 `video_pre_cond = video_expert.pre_dit(x=latents_subgoal, ...)`，然后按现有方式 prefill video cache。
- `infer_joint()` 的返回可以第一版继续返回 action；`video` 返回需要谨慎：subgoal latent 解码出来不是完整视频。建议：
  - `return_subgoal_video=false` 默认不返回 video，或只在 debug 下返回 `subgoal_video`。
  - 保持 `infer_action()` 输出 `{"action": Tensor}` 不变。

### 4. Stage 2 action horizon 保持不变

- `latents_action` 仍为 `[B, H, action_dim]`。
- Action denoise 循环不改 horizon。
- MoT attention mask 需要允许 action tokens attend 全部 K 个 subgoal tokens。
- `video_seq_len` 会自然变短，因为 `video_pre_cond["tokens"]` 来自 K 个 latent。

### 5. Goal token 接入

沿用现有 goal token bank：

- 在 proprio append 前从 raw T5 context/context_mask 编码 goal token。
- Stage 1 `video_expert(..., extra_context_emb=video_goal_hidden)` 保持启用。
- Stage 2 的 `video_pre_cond` 也传入 `extra_context_emb=video_goal_hidden`，使 subgoal-side KV cache 带 goal-conditioned context。
- 第一版不改 `ActionDiT.pre_dit()`，避免 action expert 直接依赖 goal token，先验证 `goal -> subgoal latent -> action`。

## Useful Engineering Tricks

### Trick 1: GT subgoal teacher forcing for action branch

训练时 Stage 2 action 分支优先使用 GT subgoal latent 构建 video KV cache，而不是 noisy/predicted subgoal latent。这样 action loss 的学习目标更稳定。

可配置：

```yaml
teacher_forcing_ratio: 1.0
```

后续可退火到混合 predicted subgoal：

```text
early: 100% GT subgoal
later: p% GT + (1-p)% predicted/noisy subgoal
```

### Trick 2: Subgoal token dropout

训练 Stage 2 时随机 mask/drop 一部分 subgoal tokens 或整段 subgoal latent，让 action decoder 不依赖某一个关键 token，提升鲁棒性。

建议第一版轻量实现：在 `video_pre_cond["tokens"]` 级别做 dropout，而不是改 raw latent。

```yaml
subgoal_token_dropout: 0.05
```

### Trick 3: Goal token dropout

训练 Stage 1 时以小概率不传 `video_goal_hidden`，避免模型过度依赖 alignment checkpoint，也保留 language/context 本身的能力。

```yaml
goal_token_dropout: 0.10
```

实现方式：training only，将 `video_goal_hidden=None` 或将 goal hidden 置零。第一版建议直接置零，shape 不变，便于调试 mask 长度。

### Trick 4: Action time-to-subgoal hint

K 个 subgoal 对 H 个 action step 是稀疏条件。Action 侧需要知道每个 action step roughly 对应哪个 subgoal 阶段。

第一版低风险做法：给 action tokens 增加一个简单的 subgoal progress embedding 或 timestep bias：

```text
action step i -> progress i / (H-1)
subgoal index j -> progress j / (K-1)
```

如果不想改 ActionDiT，先在 MoT/action pre-state 里加可选 time embedding；若工程成本偏高，第一版可以只记录 TODO，不阻塞主实验。

### Trick 5: Loss scaling for shorter subgoal branch

K 变小后 video/subgoal loss 的元素数量变少，`loss_video` 和 `loss_action` 的相对权重会变。建议显式记录并可配置：

```yaml
loss:
  lambda_subgoal: 1.0
  lambda_action: 1.0
```

实现时可以复用 `lambda_video`，但日志里改名为 `loss_subgoal`，避免误读。

### Trick 6: Debug decode only

Subgoal latent 不是完整视频，不建议默认 decode 成 video 评估。可以加 debug 开关：

```yaml
return_subgoal_video: false
```

用于人工检查 K 个子目标是否覆盖关键阶段。

## Validation Plan

### Static / Unit Checks

- `python -m py_compile` 覆盖：
  - `fastwam_idm.py`
  - `fastwam.py`
  - `runtime.py`
  - 新增 subgoal helper 文件（若有）
- 小张量测试 `select_subgoal_indices()`：
  - `T=15,K=5` 返回合法升序索引。
  - `K=1`、`K=T`、`K>T` 边界行为明确。
  - `custom_indices` 越界时报清晰错误。

### Training Smoke Test

- 用 1 个 mini-batch 调 `FastWAMIDM.training_loss()`：
  - `subgoal_latent.enabled=false` 与旧路径 shape 不变。
  - `enabled=true,K=5` 时 Stage 1 latent time length 为 K。
  - action loss 输出 shape 仍对应 action horizon H。
  - `loss_dict` 包含 `loss_subgoal` / `loss_action`，或兼容旧 key。

### Inference Smoke Test

- `infer_action()` 小步数测试：
  - `latents_subgoal.shape[2] == K`。
  - `latents_action.shape[1] == H`。
  - `video_seq_len` 比完整 chunk 路径缩短。
  - 输出接口至少保持 `{"action": Tensor}`。

### Ablations

第一轮建议只做这些对比：

1. Baseline IDM：完整 latent chunk length H。
2. Subgoal IDM：K=5，uniform_endpoints，无 dropout。
3. Subgoal IDM：K=5，加 goal_token_dropout + subgoal_token_dropout。
4. Subgoal IDM：K=3 / K=7，看 bottleneck 强度。

## Explicit Non-goals for V1

- 不做 learned selector。
- 不推理时完整生成 H 个 latent 再挑 K 个。
- 不直接改 ActionDiT 让 action cross-attend raw goal token。
- 不第一版修改 dataset 格式；仍从完整 video chunk 中抽 supervision。
- 不把 subgoal decode 成完整 video 作为默认输出。

## Future Innovation Directions

### Learned temporal selector

训练时从 GT full latent chunk 中学习 K 个 task-dependent supervision targets：

```text
goal_token + current_obs + language -> alpha[K, T]
target_subgoal[k] = sum_t alpha[k, t] * z_t
```

推理时模型仍只生成 K 个 subgoal latent，不生成完整 chunk。

### Goal-subgoal alignment loss

让 pooled subgoal representation 与 goal token / language embedding 做 contrastive alignment：

```text
positive: same task goal token
negative: other task goal token
```

目标是让 subgoal latent 更像任务阶段表示，而不是低层视频压缩。

### Action consistency auxiliary head

增加辅助头：

```text
current obs + GT/pred subgoals -> action chunk
```

让 subgoal latent 被迫保留对动作有用的信息。

### Adaptive K

不同任务复杂度需要不同数量子目标。后续可以让模型预测 stop token 或 confidence，支持动态 K。

## Assumptions

- 第一版在 `/data/ztk/code/iclr` 修改。
- 当前 dataset 仍能提供完整 video/action chunk。
- VAE latent time length 可能小于原始 frame/action horizon，因此 subgoal index 基于 latent time 维度选择。
- Goal token bank 已接入 IDM，并可通过 `goal_token.enabled` 控制。
- 主要验证指标应是 action success / action loss / rollout success，而不是 subgoal decode 视频质量。
