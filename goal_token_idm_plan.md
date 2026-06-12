# 将 Goal Token Bank 接入 FastWAMIDM（目标目录：`/data/ztk/code/iclr`）

## Summary

在 `iclr` 副本中把已训练的 goal token bank 接入 `FastWAMIDM`，用于 **IDM Stage 1 视频/子目标 latent 生成**。goal token 作为独立 goal-space 条件，不伪装成 T5 raw embedding；它会通过 `video_goal_adapter` 映射到 video expert hidden dim 后，拼到 video expert 的 cross-attention context 末尾，并对所有 video/subgoal query 全局可见。Action 阶段不直接使用 goal token，只通过 Stage 1 生成的视频/subgoal latent 间接受益。

## Key Changes

- 新增可复用 goal token 模块：
  - 在 `src/fastwam/models/wan22/` 下加入 goal token bank encoder，结构与当前 alignment checkpoint 兼容。
  - 支持从 alignment checkpoint 加载 text/image/query/FFN/norm 权重；推理和 IDM 训练时只调用 `encode_text(context, context_mask)`。
  - 默认冻结 goal encoder 参数；只训练/加载 `video_goal_adapter` 是否可训练由配置决定，默认训练时可训练、推理随 checkpoint 加载。

- 扩展 `WanVideoDiT` context 接口：
  - 给 `pre_dit()` 和 `forward()` 增加可选参数：
    ```python
    extra_context_emb: Optional[torch.Tensor] = None  # [B, M, hidden_dim]
    ```
  - 在 `context = self.text_embedding(context)` 后 append `extra_context_emb`。
  - 对 mask 追加全 True goal mask，并在后续扩展成 `[B, S, L+M]`。
  - 该参数默认为 `None`，不改变现有模型行为。

- 集成到 `FastWAMIDM`：
  - 初始化时增加：
    ```text
    goal_token_encoder
    video_goal_adapter: Linear(goal_dim -> video_hidden_dim)
    ```
  - goal token 从 **T5 context + context_mask** 编码，且在 proprio token append 之前计算，保持 task-level 语义，不混入当前 proprio。
  - `training_loss()` 中，`video_pre_noisy` 和 `video_pre_cond` 都传入 `extra_context_emb=video_goal_hidden`。
  - `infer_joint()` Stage 1 的 `video_expert(...)` 传入 `extra_context_emb=video_goal_hidden`。
  - Stage 2 action 去噪不直接传 goal token；`video_pre_cond` 是否带 goal hidden保持与 Stage 1 一致，但 action expert context 不追加 goal。

## Public Config / Interface Additions

- 在 IDM model config 增加可选块：
  ```yaml
  goal_token:
    enabled: true
    checkpoint_path: runs/goal_alignment/alignment_full_libero_goalspace/best.pt
    goal_dim: 512
    num_goal_tokens: 4
    hidden_dim: 2048
    num_heads: 8
    freeze_encoder: true
    train_video_adapter: true
  ```
- `create_fastwam_idm()` 和 `FastWAMIDM.from_wan22_pretrained()` 增加 `goal_token_config: dict | None = None`。
- `FastWAMIDM.infer_joint()` / `infer_action()` 不新增用户必填参数；goal token 从现有 `prompt` 或 `context/context_mask` 自动生成。
- Checkpoint 保存/加载应包含 goal encoder 和 `video_goal_adapter` 权重；旧 checkpoint 在 `goal_token.enabled=false` 时保持兼容。

## Implementation Details

- 目标目录固定为 `/data/ztk/code/iclr`。
- goal encoder checkpoint 加载规则：
  - 读取 `checkpoint["config"]` 校验 `goal_dim/num_goal_tokens/num_heads`。
  - 加载 `checkpoint["model"]` 到 goal token encoder。
  - 若配置与 checkpoint 不一致，启动时报明确错误。
- `video_goal_adapter` 输出 dtype/device 与 video expert context hidden 对齐。
- goal hidden mask 始终全局可见：
  ```python
  goal_mask = torch.ones(B, M, dtype=torch.bool, device=context_mask.device)
  ```
- 训练时 goal token 参与 video/subgoal branch 梯度流：
  - 若 `freeze_encoder=true`，只训练 adapter 和原 IDM 可训练参数。
  - 若 `train_video_adapter=true`，adapter 加入 optimizer；否则 adapter 仅随 checkpoint/初始化固定。
- 不改 `ActionDiT.pre_dit()`，避免 action 直接依赖 goal token，保持结构：
  ```text
  goal -> Stage 1 video/subgoal latent -> Stage 2 action
  ```

## Test Plan

- 静态/单元检查：
  - `python -m py_compile` 覆盖新增 goal 模块、`fastwam_idm.py`、`wan_video_dit.py`、runtime。
  - 小张量测试 `WanVideoDiT.pre_dit(extra_context_emb=...)`：
    - context length 从 `L` 变为 `L+M`
    - context_mask 最后一段全 True
    - `extra_context_emb=None` 输出与旧路径形状一致。
  - 小张量测试 `FastWAMIDM` goal helper：
    - prompt/context 输入均能产生 `[B, M, goal_dim]`
    - adapter 输出 `[B, M, video_hidden_dim]`

- 训练路径验证：
  - 用 1 个 mini-batch 调 `FastWAMIDM.training_loss()`，确认无 shape/device/dtype 错误。
  - 检查 loss_dict 不变，现有 trainer 无需改调用。
  - 验证 `goal_token.enabled=false` 时训练路径与原 IDM 一致。

- 推理路径验证：
  - 用 `infer_action()` 小步数 smoke test，确认 Stage 1 video denoise 和 Stage 2 action denoise均正常。
  - 对比 `goal_token.enabled=false/true` 输出接口完全一致：
    ```python
    {"action": Tensor}
    {"video": list[Image], "action": Tensor}
    ```

## Assumptions

- 使用 `/data/ztk/code/iclr` 作为唯一修改目标。
- 第一版 goal token 只服务 IDM Stage 1，不直接接入 action expert。
- 使用当前健康 checkpoint 形态：`num_goal_tokens=4`、`goal_dim=512`；若实际 checkpoint 不同，以配置显式值为准并在加载时校验。
- goal token 是 task-level 终态语义条件；episode-level 几何仍由 current observation / video latent 提供。
