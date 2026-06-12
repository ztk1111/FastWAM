# Bidirectional IDM Training 计划（当前实施版）

## Scope

本版本只针对 `FastWAMIDM`：

- 只改 IDM 的训练路径。
- 不改 `FastWAM` / `FastWAMJoint` / uncond joint training 的语义。
- 推理仍只走正向 `infer_action()` / `infer_joint()`。
- goal token 先不用，runtime 中强制无效化为 `None`。
- subgoal latent 路径删除；当前 IDM 本身已经是压缩 video latent 条件动作，继续做 subgoal 不再作为当前方向。

当前目标是用同一样本的正反向 paired training 给 IDM 加一个动力学一致性的辅助监督。

## Core Idea

对每个训练样本，同时训练两个方向：

```text
Forward:
  video:  [v0, v1, ..., vT]
  action: [a0, a1, ..., aH]
  direction: forward

Backward:
  video:  [vT, ..., v1, v0]
  action: reverse_action([a0, ..., aH])
  direction: backward
```

训练 loss：

```text
L = L_forward + lambda_backward * L_backward
```

第一版不加 cycle loss，不做额外 rollout，只复用 IDM 原本的 video/action diffusion loss。

## Direction Token

正反向标记保留，但只在 `bidirectional.enabled=true` 时启用。默认关闭 bidirectional 时，不会额外拼 direction token，因此 baseline IDM 行为不变。

启用后：

```text
context = [direction_token, text_context, optional proprio_tokens]
```

方向 id：

```text
0 = forward
1 = backward
```

推理固定使用 `forward`。

## Reverse Action

反向 action 不是简单 flip。当前 helper 支持可选 delta mask：

```python
action_rev = action.flip(dims=[1]).clone()
action_rev[..., delta_mask] *= -1
```

如果 `bidirectional.delta_action_dim_mask=null`，则只做时间反转，不翻符号。

LIBERO 可用：

```yaml
bidirectional:
  delta_action_dim_mask: [true, true, true, true, true, true, false]
```

RoboTwin 如果 action 是 absolute 或暂时不确定，先保持 `null`。

## Current Config

`configs/model/fastwam_idm.yaml`：

```yaml
goal_token:
  enabled: false

bidirectional:
  enabled: false
  paired_same_sample: true
  lambda_backward: 1.0
  direction_token: true
  reverse_action_mode: delta_aware
  delta_action_dim_mask: null
```

开启训练时通过 Hydra override：

```bash
model.bidirectional.enabled=true
model.bidirectional.lambda_backward=1.0
```

LIBERO 可额外 override delta mask。

## Implementation Summary

`FastWAMIDM.training_loss()` 现在是 wrapper：

```text
if bidirectional disabled:
  run single forward IDM loss

if bidirectional enabled:
  run forward single-direction IDM loss
  construct backward sample from same batch
  run backward single-direction IDM loss
  return forward + lambda_backward * backward
```

单向 loss 主体保留原 IDM teacher-forcing 结构：

```text
noisy video branch + condition video branch + noisy action branch
```

区别是启用 bidirectional 时，context 前面会拼 direction token。

## Logging

启用后会记录：

```text
loss_video_forward
loss_action_forward
loss_total_forward
loss_video_backward
loss_action_backward
loss_total_backward
loss_video
loss_action
loss_bidirectional_total
```

其中 `loss_video` / `loss_action` 是正反平均，方便兼容原 trainer。

## Inference

推理不走反向：

```text
direction = forward
```

goal token 不传入 video expert，`extra_context_emb=None`。

## Recommended First Run

先小 batch：

```bash
batch_size=4 \
model.bidirectional.enabled=true \
model.bidirectional.lambda_backward=1.0
```

如果显存吃紧：

```bash
model.bidirectional.lambda_backward=0.5
```

但注意 lambda 只降梯度权重，不省显存；省显存主要靠 batch size / checkpointing。

## Validation

最小检查：

- `bidirectional.enabled=false` 时，代码走单向 forward，且不添加 direction token。
- `enabled=true` 时，同一个 batch 会跑 forward 和 backward 两次。
- backward video 第一帧是原始最后一帧。
- backward action 在 delta mask 维度翻符号。
- `direction_embedding` 进入 optimizer，并随 checkpoint 保存/加载。
- 推理接口不变。

## Research Story

> We introduce paired bidirectional IDM training to improve temporal dynamics consistency. For each trajectory chunk, the same IDM is trained in both the forward goal-reaching direction and the backward goal-regressing direction, conditioned by a learned direction token. The backward objective provides an auxiliary self-supervised signal for learning reversible latent-action dynamics, while inference remains unchanged and uses only the forward direction.
