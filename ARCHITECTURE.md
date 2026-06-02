# FastWAM 代码库架构文档

> **FastWAM** (Fast World-Action Model) 是基于 Wan2.2-TI2V-5B 视频生成模型构建的世界-动作联合模型，采用 **MoT (Mixture of Transformers)** 混合注意力架构，同时预测未来视频帧和机器人动作。

---

## 目录

1. [项目概览](#1-项目概览)
2. [目录结构](#2-目录结构)
3. [核心架构：MoT 混合注意力](#3-核心架构mot-混合注意力)
4. [模型骨干详解](#4-模型骨干详解)
5. [数据流全景](#5-数据流全景)
6. [训练流程](#6-训练流程)
7. [推理流程](#7-推理流程)
8. [配置文件体系](#8-配置文件体系)
9. [关键维度对照表](#9-关键维度对照表)

---

## 1. 项目概览

### 1.1 三种模型变体

| 变体 | 类名 | 文件 | 说明 |
|------|------|------|------|
| **FastWAM** (默认) | `FastWAM` | `fastwam.py` | 动作仅关注视频第一帧，用于 action-conditioned 视频生成 |
| **FastWAMJoint** | `FastWAMJoint` | `fastwam_joint.py` | 动作关注全部视频 token，联合去噪 |
| **FastWAMIDM** | `FastWAMIDM` | `fastwam_idm.py` | IDM (Inverse Dynamics Model) 变体，teacher-forcing 两阶段推理 |

### 1.2 核心思想

```
输入: 首帧图像 + 语言指令 + 本体感知(proprioception)
     ↓
输出: 未来 N 帧视频 + 动作序列 (action horizon)
     ↓
机制: 视频DiT和动作DiT共享Transformer层，通过MoT混合注意力交互
```

---

## 2. 目录结构

```
FastWAM/
├── scripts/                          # 入口脚本
│   ├── train.py                      # 训练入口 (Hydra)
│   ├── preprocess_action_dit_backbone.py  # 预处理ActionDiT权重
│   ├── precompute_text_embeds.py     # 预计算文本嵌入缓存
│   └── accelerate_configs/          # DeepSpeed/Accelerate 配置
│
├── src/fastwam/                      # 核心库
│   ├── runtime.py                    # 运行时工厂函数 + 训练/推理调度
│   ├── trainer.py                    # Wan22Trainer 训练循环
│   ├── models/wan22/
│   │   ├── fastwam.py               # FastWAM 主模型 (父类)
│   │   ├── fastwam_joint.py         # FastWAMJoint 联合变体
│   │   ├── fastwam_idm.py           # FastWAMIDM 两阶段变体
│   │   ├── wan22.py                 # Wan22Core 纯视频基类
│   │   ├── action_dit.py            # ActionDiT 动作专家
│   │   ├── wan_video_dit.py         # WanVideoDiT 视频专家 + 基础组件
│   │   ├── mot.py                   # MoT 混合注意力核心
│   │   ├── schedulers/
│   │   │   └── scheduler_continuous.py  # Flow Matching 连续时间调度器
│   │   └── helpers/
│   │       ├── loader.py            # 模型下载/加载
│   │       ├── io.py                # 状态字典读写 + 文件哈希
│   │       ├── gradient.py          # 梯度检查点工具
│   │       └── state_dict_converters.py  # 权重命名映射
│   ├── datasets/lerobot/            # LeRobot 数据集适配层
│   └── utils/                       # 通用工具 (视频IO/指标/采样器等)
│
├── configs/                          # Hydra 配置
│   ├── model/                        # 模型配置 (fastwam / fastwam_idm / fastwam_joint)
│   ├── data/                         # 数据配置 (libero / robotwin)
│   ├── task/                         # 任务配置
│   ├── train.yaml                    # 训练超参
│   └── sim_*.yaml                    # 仿真评估配置
│
└── experiments/                      # 实验评估
    ├── libero/                       # LIBERO benchmark
    └── robotwin/                     # RoboTwin benchmark
```

---

## 3. 核心架构：MoT 混合注意力

### 3.1 架构总览

```
                    ┌──────────────────────────────┐
                    │         MoT Forward           │
                    │  (mot.py, 逐层混合注意力)      │
                    │                              │
  Video Tokens ────►│  Layer 0: Mixed Self-Attn   ├────► Video Output
  [B, Sv, Dv]       │    Q,K,V = cat(video,action) │      [B, Sv, Dv]
                    │    attn_mask: 联合掩码        │
  Action Tokens ───►│    split → post-block 各自处理│────► Action Output
  [B, Sa, Da]       │                              │      [B, Sa, Da]
                    │  Layer 1..29: 同上           │
                    └──────────────────────────────┘
```

每层 Transformer 中，两个专家的 Q/K/V 在序列维度上拼接后做统一的 Flash Attention，attention 输出再按序列长度切回给各自的 post-block（含 cross-attn 和 FFN）。

### 3.2 注意力掩码规则

```
┌─────────────────────┬──────────────┬──────────────┐
│                     │ Video Key    │ Action Key   │
├─────────────────────┼──────────────┼──────────────┤
│ Video Query         │ 因果/双向     │ 全 False     │
│ Action Query        │ 首帧 True     │ 全 True      │
│    (FastWAM 默认)   │ (其余 False)  │              │
│ Action Query        │ 全 True       │ 全 True      │
│    (FastWAMJoint)   │              │              │
└─────────────────────┴──────────────┴──────────────┘
```

### 3.3 三个变体的注意力差异

| 特性 | FastWAM | FastWAMJoint | FastWAMIDM |
|------|---------|-------------|------------|
| Action→Video attention | 仅首帧 token | 全部 video token | 仅 cond_video token |
| 推理方式 | joint 同时去噪 | joint 同时去噪 | 先 video 后 action 两阶段 |
| video_dit action_conditioned | True | False | False |
| 训练时 GT action 输入 | 是 | N/A | N/A (teacher forcing) |

---

## 4. 模型骨干详解

### 4.1 WanVideoDiT（视频专家）

**用途**: 基于 Wan2.2-TI2V-5B 预训练权重的视频扩散 Transformer

```
输入数据流:
  x (噪声潜变量)     [B, C=48, T_latent, H//8, W//8]  # VAE 压缩后
  timestep           [B]
  context            [B, L_text, 4096]                # T5 文本嵌入
  action (可选)      [B, action_horizon, 7]           # GT 动作条件

内部处理:
  1. Patch Embedding: Conv3d(48→hidden_dim, kernel=(1,2,2), stride=(1,2,2))
     → [B, hidden_dim, T_latent, H//16, W//16]
  2. Tokenize: rearrange → [B, T_latent * H_patch * W_patch, hidden_dim]
     例: latent shape [1,48,5,28,28] → [1, 5*14*14=980, 3072] (video expert)
  3. Text Embedding: Linear(4096→hidden_dim) + GELU + Linear(hidden_dim→hidden_dim)
  4. Time Embedding: sinusoidal_1d → Linear + SiLU → time_projection(6*hidden_dim)
  5. 30× DiTBlock (详见下)
  6. Head: LayerNorm → Linear → unpatchify → [B, C, T_latent, H//8, W//8]

关键配置 (video expert):
  hidden_dim=3072, ffn_dim=14336, num_heads=24, attn_head_dim=128
  num_layers=30, text_dim=4096, freq_dim=256
```

### 4.2 DiTBlock（Transformer 基础块）

**视频和动作专家共用同一结构**，每块包含：

```
DiTBlock 内部流程 (维度示例: hidden_dim=3072 或 1024):

  x [B, S, D]
  │
  ├─ 1. Self-Attention:
  │   modulation(6,D) → shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
  │   norm1(x) → modulate(x, shift_msa, scale_msa) → Q/K/V投影 → RMSNorm → RoPE
  │   → Flash Attention(Q,K,V) → gate(x, gate_msa, attn_out)
  │
  ├─ 2. Cross-Attention:
  │   norm3(x) → Q投影+RMSNorm / 对 context 做 K,V投影+RMSNorm
  │   → Flash Attention → 残差加回
  │
  └─ 3. FFN:
      norm2(x) → modulate(x, shift_mlp, scale_mlp)
      → Linear(D→ffn_dim) → GELU → Linear(ffn_dim→D)
      → gate(x, gate_mlp, ffn_out)
```

### 4.3 ActionDiT（动作专家）

**用途**: 动作去噪专家，结构与视频专家类似但 hidden_dim 更小

```
输入数据流:
  action_tokens       [B, action_horizon, action_dim]  # 例: [1, 8, 7]
  timestep            [B]
  context             [B, L_text, 4096]

内部处理:
  1. Action Encoder:  Linear(action_dim → hidden_dim)
     → [1, 8, 7] → [1, 8, 1024]
  2. Text Embedding:  Linear(4096→1024) + GELU + Linear(1024→1024)
  3. Time Embedding:  sinusoidal → Linear + SiLU → time_projection(6*1024)
  4. 30× DiTBlock:    与视频专家结构相同，但 hidden_dim=1024, ffn_dim=4096
  5. ActionHead:      LayerNorm → Linear(1024→action_dim)
     → [1, 8, 1024] → [1, 8, 7]

关键配置 (action expert):
  hidden_dim=1024, ffn_dim=4096, num_heads=24, attn_head_dim=128
  num_layers=30, text_dim=4096, freq_dim=256, action_dim=7
```

**权重初始化策略**:
- 从预训练 WanVideoDiT 的 backbone 权重通过 **1D 线性插值** 迁移（preprocess_action_dit_backbone.py）
- hidden_dim 3072→1024, ffn_dim 14336→4096 通过 F.interpolate 做最后一维缩放
- num_heads/attn_head_dim/num_layers 必须与视频专家一致（否则 MoT 混合注意力无法工作）
- Alpha 缩放: 插值后乘以 sqrt(3072/1024) 补偿方差变化

### 4.4 MoT（混合注意力容器）

**用途**: 逐层执行视频+动作的混合自注意力，管理 Q/K/V 拼接和输出切分

```
forward() 核心循环:

  for layer_idx in range(num_layers):       # 30 层
    1. 收集各专家 Q/K/V:
       for expert in [video, action]:
         block = expert.blocks[layer_idx]
         modulate(norm1(x), shift, scale) → q,k,v → RMSNorm → RoPE

    2. 拼接:
       q_cat = cat([q_video, q_action], dim=1)   # [B, Sv+Sa, H*Dh]
       k_cat = cat([k_video, k_action], dim=1)
       v_cat = cat([v_video, v_action], dim=1)

    3. 混合注意力:
       mixed = FlashAttention(q_cat, k_cat, v_cat, attention_mask)

    4. 切分 + 各专家的 post-block:
       mixed_video = mixed[:, :Sv, :]   → cross_attn + FFN
       mixed_action = mixed[:, Sv:, :]  → cross_attn + FFN
```

**KV Cache 机制** (推理时使用):
- `prefill_video_cache()`: 预填充视频分支所有 30 层的 K/V，返回 layer-wise cache
- `forward_action_with_video_cache()`: 动作分支逐层使用缓存的视频 K/V 做混合注意力，省去视频分支重复计算

---

## 5. 数据流全景

### 5.1 训练数据流

```
Dataset (LeRobot)
  │
  ├─ video:      [B, 3, T, H, W]        原始RGB帧, T=num_frames
  ├─ action:     [B, T-1, action_dim]    动作序列
  ├─ state:      [B, T, state_dim]       本体感知 (关节角等)
  ├─ prompt:     str/list[str]           语言指令
  └─ (或 precomputed context/context_mask)
  │
  ▼ build_inputs()
  │
  ├─ VAE Encode:  video → latents [B, C, T_latent, H//8, W//8]
  │                 例: [1,3,17,224,224] → [1,48,5,28,28]
  │
  ├─ Text Embed:  prompt → T5 Encoder → context [B, L, 4096]
  │                (训练时用预计算缓存)
  │
  ├─ Proprio:     state[:,0,:] → Linear(state_dim→text_dim) → 拼入 context
  │
  ▼ training_loss()
  │
  ├─ Video Branch:
  │   noise_video ~ N(0,I) × latents
  │   timestep_video ~ FlowMatchScheduler.sample_training_t()
  │   noisy_latents = (1-σ)*latents + σ*noise_video
  │   target_video = noise_video - latents
  │   → video_expert.pre_dit() → MoT blocks → post_dit() → pred_video
  │
  ├─ Action Branch:
  │   noise_action ~ N(0,I) × action
  │   timestep_action ~ FlowMatchScheduler.sample_training_t()
  │   noisy_action = (1-σ)*action + σ*noise_action
  │   target_action = noise_action - action
  │   → action_expert.pre_dit() → MoT blocks → post_dit() → pred_action
  │
  ▼ Loss:
    loss_video = MSE(pred_video, target_video) × training_weight
    loss_action = MSE(pred_action, target_action) × training_weight
    loss_total = λ_video * loss_video + λ_action * loss_action
```

### 5.2 推理数据流（FastWAM Joint 模式）

```
用户输入: input_image [1,3,H,W] + prompt/horizon
           │
  VAE Encode: → first_frame_latents [1,C,1,H//8,W//8]
            │
  初始化:   latents_video [1,C,T_latent,H//8,W//8] ~ N(0,I)
            latents_action [1,horizon,action_dim] ~ N(0,I)
            latents_video[:,:,0:1] = first_frame_latents  # 固定首帧
            │
  Text Encode: prompt → context [1,L,4096] + mask
            │
  for step in inference_steps:  (如 20 步)
    ├─ video_expert.pre_dit(latents_video, t_video, context) → video_tokens [1,Sv,D]
    ├─ action_expert.pre_dit(latents_action, t_action, context) → action_tokens [1,Sa,D]
    ├─ MoT(video_tokens, action_tokens, attn_mask)
    ├─ video_expert.post_dit(mixed_video) → pred_video [1,C,T_latent,H//8,W//8]
    ├─ action_expert.post_dit(mixed_action) → pred_action [1,horizon,7]
    ├─ latents_video = scheduler.step(pred_video, delta, latents_video)
    ├─ latents_action = scheduler.step(pred_action, delta, latents_action)
    └─ latents_video[:,:,0:1] = first_frame_latents  # 保持首帧
            │
  VAE Decode: latents_video → video_frames [T, H, W, 3]
  动作输出:   latents_action → denormalize → 机器人关节角序列
```

### 5.3 推理数据流（FastWAMIDM 两阶段模式）

```
Stage 1 (纯视频去噪):
  video_expert.forward(latents_video, t, context, action=None)
  → 逐步去噪得到干净视频潜变量

Stage 2 (动作去噪，使用缓存视频KV):
  video_expert.pre_dit(denoised_latents, t=0, context)
  → mot.prefill_video_cache()  # 缓存全部30层视频K/V

  for step in inference_steps:
    action_expert.pre_dit(latents_action, t, context) → action_tokens
    mot.forward_action_with_video_cache(action_tokens, video_kv_cache)
    → action_expert.post_dit() → scheduler.step()
```

---

## 6. 训练流程

### 6.1 入口

```bash
# 单卡训练
python scripts/train.py model=fastwam data=libero_2cam ...

# 多卡分布式
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
    scripts/train.py model=fastwam data=libero_2cam ...
```

### 6.2 训练循环 (Wan22Trainer)

```
初始化:
  ├─ Accelerator(混合精度 + 梯度累积 + DeepSpeed ZeRO)
  ├─ 冻结非 DiT 模块 (VAE, text_encoder 等)
  ├─ 仅 DiT + proprio_encoder 可训练
  ├─ AdamW (lr, weight_decay, betas=(0.9, 0.95))
  └─ CosineAnnealingLR + 5% warmup

每步循环:
  while global_step < max_steps:
    ├─ 从 DataLoader 取 sample
    ├─ accelerator.accumulate():
    │   ├─ autocast(): model.training_loss(sample) → loss
    │   ├─ accelerator.backward(loss)
    │   └─ 梯度同步后:
    │       ├─ clip_grad_norm_(max_grad_norm)
    │       ├─ optimizer.step()
    │       ├─ scheduler.step()
    │       └─ optimizer.zero_grad()
    ├─ 日志记录 (loss, grad_norm, lr, speed)
    ├─ 定期评估: evaluate() → val_loss, PSNR, SSIM, action_L1/L2
    └─ 定期保存: weights/{step}.pt + accelerate state
```

### 6.3 评估指标

| 指标 | 含义 |
|------|------|
| `psnr_rg` | 生成视频 vs GT 视频的 PSNR |
| `ssim_rg` | 生成视频 vs GT 视频的 SSIM |
| `psnr_rd` | 生成视频 vs VAE 重建视频的 PSNR |
| `ssim_rd` | 生成视频 vs VAE 重建视频的 SSIM |
| `psnr_dg` | VAE 重建 vs GT 的 PSNR (VAE 重建上界) |
| `ssim_dg` | VAE 重建 vs GT 的 SSIM (VAE 重建上界) |
| `action_l1` | 预测动作 vs GT 动作的 L1 距离 |
| `action_l2` | 预测动作 vs GT 动作的 L2 距离 |

---

## 7. 推理流程

### 7.1 策略部署 (deploy_policy.py)

```
WorldActionRobotWinPolicy:
  ┌─────────────────────────────────────┐
  │  step(task_env, observation):       │
  │    if 动作队列为空:                  │
  │      推理生成 action_chunk [T, D]    │
  │      按 replan_steps 入队           │
  │    pop 队列首动作 → take_action()   │
  └─────────────────────────────────────┘
```

### 7.2 推理流水线

```
观察 → resize/crop 相机图像 → concat 多视图 → 归一化到 [-1,1]
     → normalize(proprio)
     → model.infer_action(prompt, image, proprio, horizon)
     → denormalize(action) → 执行
```

---

## 8. 配置文件体系

### 8.1 配置继承关系

```
configs/train.yaml           # 训练超参 (lr, batch_size, epochs, ...)
  └─ configs/model/*.yaml    # 模型定义 (fastwam / fastwam_idm / fastwam_joint)
       ├─ video_dit_config    # 视频 DiT 结构参数
       └─ action_dit_config   # 动作 DiT 结构参数
  └─ configs/data/*.yaml     # 数据配置 (libero_2cam / robotwin)
  └─ configs/task/*.yaml     # 任务特定参数
```

### 8.2 关键模型配置 (fastwam.yaml)

```yaml
_target_: fastwam.runtime.create_fastwam
model_id: Wan-AI/Wan2.2-TI2V-5B           # 预训练视频模型
tokenizer_model_id: Wan-AI/Wan2.1-T2V-1.3B
load_text_encoder: false                   # 训练时用预计算缓存
action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
skip_dit_load_from_pretrain: false
mot_checkpoint_mixed_attn: true            # MoT 层梯度检查点

video_dit_config:
  hidden_dim: 3072                          # 视频专家隐藏维度
  ffn_dim: 14336                           # FFN 扩展维度
  num_heads: 24                            # 注意力头数 (必须与 action 一致)
  attn_head_dim: 128                       # 每头维度 (必须与 action 一致)
  num_layers: 30                           # Transformer 层数 (必须与 action 一致)
  text_dim: 4096                           # T5 文本嵌入维度
  freq_dim: 256                            # 时间嵌入频率维度
  action_conditioned: true                 # 视频专家接受 GT action 条件

action_dit_config:
  hidden_dim: 1024                         # 动作专家隐藏维度 (更小)
  ffn_dim: 4096                            # FFN 扩展维度 (更小)
  num_heads: 24                            # = video expert
  attn_head_dim: 128                       # = video expert
  num_layers: 30                           # = video expert
  text_dim: 4096                           # = video expert
  freq_dim: 256                            # = video expert
```

---

## 9. 关键维度对照表

### 9.1 模型参数规模

| 组件 | 参数量 | bf16显存 | 说明 |
|------|--------|---------|------|
| WanVideoDiT (5B) | ~5.0B | ~10 GB | 视频生成骨干 |
| ActionDiT | ~0.4B | ~0.8 GB | 动作预测骨干 (hidden_dim=1024) |
| T5-XXL Text Encoder | ~11B | ~22 GB | 文本编码器 (训练时用缓存，不加载) |
| WanVideoVAE38 | ~0.1B | ~0.2 GB | 视频压缩/解压 |
| MoT (无参数) | 0 | 0 | 仅做混合注意力调度 |
| **训练总计** | ~5.5B | ~13-15 GB | DiT + ActionDiT + VAE (不含T5) |

### 9.2 张量维度速查

| 张量 | 形状 | 说明 |
|------|------|------|
| 原始视频 (pixel) | `[B, 3, T, H, W]` | 例: `[1, 3, 17, 224, 224]` |
| VAE 潜变量 | `[B, 48, T_lat, H//8, W//8]` | 例: `[1, 48, 5, 28, 28]` |
| 视频 token 序列 | `[B, T_lat·H_p·W_p, 3072]` | 例: `[1, 980, 3072]` |
| 动作序列 | `[B, horizon, 7]` | 例: `[1, 8, 7]` (7DoF) |
| 动作 token 序列 | `[B, horizon, 1024]` | 例: `[1, 8, 1024]` |
| 文本嵌入 | `[B, L, 4096]` | 例: `[1, 128, 4096]` |
| MoT 拼接序列 | `[B, Sv+Sa, D]` | 例: `[1, 980+8=988, 3072]` (隐式广播) |
| 混合注意力掩码 | `[Sv+Sa, Sv+Sa]` | 例: `[988, 988]` (bool) |
| RoPE 频率 | `[max_seq_len, attn_head_dim//2]` | 例: `[1024, 64]` (complex) |
| Flow 时间步 | `[B]` | 例: `[1]` (0~1000 连续值) |
| t_mod 调制 | `[B, 6, D]` 或 `[B, S, 6, D]` | 6通道: msa_shift/scale/gate + mlp_shift/scale/gate |

### 9.3 数据预处理流水线

```
原始传感器数据
  ↓ LerobotDataset.__getitem__()
  ├─ video: 读取 MP4 → decode → 归一化 [0,1] → resize → 转 Tensor [3,T,H,W]
  ├─ action: absolute → relative (可选) → 归一化 (用数据集统计量)
  ├─ state: 合并关节角+末端位姿 → 归一化
  └─ prompt: 语言指令 (训练时预先 T5 编码为 context/mask 缓存)
```

---

## 附录 A: 关键函数签名

### A.1 FastWAM.training_loss

```
输入: sample (dict)
  sample["video"]:         [B, 3, T, H, W]      原始视频帧
  sample["action"]:        [B, T-1, action_dim]  动作序列
  sample["context"]:       [B, L, 4096]          T5 文本嵌入
  sample["context_mask"]:  [B, L]                文本掩码 (bool)
  sample["proprio"]:       [B, T, proprio_dim]   本体感知 (可选)
  sample["action_is_pad"]: [B, T-1]              动作 padding 掩码 (可选)
  sample["image_is_pad"]:  [B, T]                图像 padding 掩码 (可选)

输出:
  loss_total: 标量, λ_video * MSE(video) + λ_action * MSE(action)
  loss_dict:  {"loss_video": float, "loss_action": float}
```

### A.2 WanVideoDiT.forward

```
输入:
  x:                       [B, C, T_lat, H, W]   VAE潜变量
  timestep:                [B]                    时间步
  context:                 [B, L, 4096]           文本嵌入
  context_mask (可选):     [B, L]                 文本掩码
  action (可选):           [B, A_horizon, 7]      GT动作条件
  fuse_vae_embedding_in_latents: bool           首帧注入模式

输出:
  noise_pred:              [B, C, T_lat, H, W]   预测的噪声/向量场
```

### A.3 ActionDiT.forward

```
输入:
  action_tokens:           [B, horizon, action_dim=7]
  timestep:                [B]
  context:                 [B, L, 4096]
  context_mask (可选):     [B, L]

输出:
  pred_action:             [B, horizon, action_dim=7]
```

### A.4 MoT.forward

```
输入:
  embeds_all:  {"video": [B, Sv, Dv], "action": [B, Sa, Da]}
  attention_mask:  [Sv+Sa, Sv+Sa]  bool
  freqs_all:   {"video": [Sv, 1, rope_dim], "action": [Sa, 1, rope_dim]}
  context_all: {"video": {"context":..., "mask":...}, "action": {...}}
  t_mod_all:   {"video": [B, Sv, 6, Dv], "action": [B, 6, Da]}

输出:
  tokens_all:  {"video": [B, Sv, Dv], "action": [B, Sa, Da]}
```

---

## 附录 B: 预处理脚本流程

### preprocess_action_dit_backbone.py

```
目的: 将预训练 WanVideoDiT 的权重迁移到 ActionDiT 骨干
      通过线性插值处理维度不匹配的参数

流程:
  1. 加载 Wan2.2-TI2V-5B 视频 DiT (hidden_dim=3072)
  2. 创建 ActionDiT (hidden_dim=1024)
  3. 逐参数匹配:
     - 形状相同 → 直接拷贝 (如 attention Q/K/V/O 投影，因为 head 数相同)
     - 形状不同 → 1D 线性插值最后一维 (如 FFN 层)
     - 跳过 action_encoder.* 和 head.* (随机初始化)
  4. Alpha 缩放: 插值后 × sqrt(3072/1024)
  5. 保存为 .pt checkpoint
```

---

## 附录 C: 常见问题

### C.1 为什么 ActionDiT 的 num_heads/attn_head_dim/num_layers 必须与视频 DiT 一致？

MoT 混合注意力在每一层将视频和动作的 Q/K/V 在序列维度拼接后做统一的 Flash Attention，所有 token 必须在同一个注意力空间中。如果 head 数或 head 维度不同，Q 和 K 的点积维度就不匹配。

### C.2 为什么预处理脚本不需要文本编码器？

预处理只需要视频 DiT 的权重来做维度插值迁移，不需要文本编码器。如果意外加载 T5-XXL (~22GB bf16)，会在 24GB GPU 上 OOM。

### C.3 三种训练模式的选择？

- `fastwam`: 默认模式，video conditioned on GT action，适合 action-conditioned video prediction
- `fastwam_joint`: action 关注全部 video token，适合联合 video+action 生成
- `fastwam_idm`: 两阶段推理，先生成视频再基于视频预测动作，适合纯 action 预测任务
