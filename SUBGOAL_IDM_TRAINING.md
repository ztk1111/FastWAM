# Subgoal IDM 训练启动说明

这份文档说明第一版 Subgoal FastWAMIDM 的训练启动流程。

## 训练目标

普通 IDM 路径会先 denoise 一个完整未来 observation/video latent chunk，再把 video-side MoT KV cache 提供给 action chunk denoising。Subgoal IDM 保持 action horizon 不变，但 Stage 1 只 denoise `K` 个 latent observation：

```text
current obs + language/proprio + optional goal token
  -> K 个 subgoal latent
  -> video-side KV cache
  -> 原 action horizon 的 action chunk
```

实现默认关闭。下面的 launcher 会通过 Hydra overrides 打开 `model.subgoal_latent.enabled=true`。

## 启动脚本

使用：

```bash
bash scripts/train_subgoal_idm_zero2.sh <nproc_per_node> [hydra_overrides...]
```

默认 task 是 `libero_idm_2cam224_1e-4`，默认 `K=5`，默认不启用 goal token bank。

单机 LIBERO 示例：

```bash
conda activate fastwam
bash scripts/train_subgoal_idm_zero2.sh 8
```

RoboTwin 示例：

```bash
TASK=robotwin_idm_3cam_384_1e-4 \
NUM_SUBGOAL_LATENTS=5 \
bash scripts/train_subgoal_idm_zero2.sh 8
```

启用 goal token bank 示例：

```bash
GOAL_TOKEN_ENABLED=true \
GOAL_TOKEN_CHECKPOINT=runs/goal_alignment/alignment/best.pt \
NUM_SUBGOAL_LATENTS=5 \
bash scripts/train_subgoal_idm_zero2.sh 8
```

额外 Hydra overrides 会继续透传给 `scripts/train_zero2.sh`：

```bash
bash scripts/train_subgoal_idm_zero2.sh 8 \
  batch_size=8 \
  learning_rate=5e-5 \
  max_steps=20000 \
  wandb.enabled=true
```

## 可用环境变量

脚本支持这些变量：

- `TASK`：Hydra task config 名，默认 `libero_idm_2cam224_1e-4`。
- `NUM_SUBGOAL_LATENTS`：Stage 1 使用的 latent steps 数量，默认 `5`。
- `SUBGOAL_STRATEGY`：`uniform_endpoints`、`uniform_inside` 或 `custom`，默认 `uniform_endpoints`。
- `INCLUDE_FIRST_LATENT`：默认 `true`。第一个 latent 是当前观测 anchor，沿用现有 video expert 的条件方式。
- `SUBGOAL_TOKEN_DROPOUT`：训练时对 teacher-forcing subgoal cache 做 token dropout，默认 `0.05`。
- `GOAL_TOKEN_DROPOUT`：训练 Stage 1 时对 goal-token 条件做 dropout，默认 `0.10`。
- `GOAL_TOKEN_ENABLED`：设为 `true` 时加载 goal token bank。
- `GOAL_TOKEN_CHECKPOINT`：`GOAL_TOKEN_ENABLED=true` 时必填。
- `WANDB_NAME`：可选的 wandb run name 覆盖。

底层仍复用 `train_zero2.sh`，所以多机变量也保持一致：

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<rank0_ip> MASTER_PORT=29500 \
bash scripts/train_subgoal_idm_zero2.sh 8
```

## 重要注意事项

`NUM_SUBGOAL_LATENTS` 统计的是 VAE latent 时间轴，不是原始 action steps。如果数据的视频 chunk 在 VAE 下采样后的 latent steps 太少，模型会直接报错；这时需要降低 `NUM_SUBGOAL_LATENTS`，或者增大视频 chunk 长度。

默认 `INCLUDE_FIRST_LATENT=true` 是有意的。当前架构中，当前观测主要通过第一个 latent anchor 进入 video expert。`K=5` 表示 1 个当前观测 anchor 加 4 个未来 subgoal latent。

Stage 2 的 action horizon 不变。缩短的只是 video/subgoal side 的 latent/token/cache 序列长度。

## 推荐第一轮实验

建议先跑这几组：

```bash
# Subgoal baseline，不启用 goal token，不加 dropout。
NUM_SUBGOAL_LATENTS=5 SUBGOAL_TOKEN_DROPOUT=0.0 GOAL_TOKEN_DROPOUT=0.0 \
bash scripts/train_subgoal_idm_zero2.sh 8

# 加轻量 dropout，提高鲁棒性。
NUM_SUBGOAL_LATENTS=5 SUBGOAL_TOKEN_DROPOUT=0.05 GOAL_TOKEN_DROPOUT=0.10 \
bash scripts/train_subgoal_idm_zero2.sh 8

# 更强 bottleneck。
NUM_SUBGOAL_LATENTS=3 \
bash scripts/train_subgoal_idm_zero2.sh 8

# 更弱 bottleneck。
NUM_SUBGOAL_LATENTS=7 \
bash scripts/train_subgoal_idm_zero2.sh 8
```

## 输出目录

脚本复用 `scripts/train_zero2.sh`，输出目录为：

```text
runs/<task>/<run_id>
```

启动前设置 `RUN_ID=<name>` 可以固定输出目录名。
