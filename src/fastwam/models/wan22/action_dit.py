"""
action_dit.py

FastWAM 项目中动作扩散 Transformer（Action DiT）模型的实现。
该模块负责在 WAN 2.2 视频生成管线中，根据文本条件和当前动作序列
预测下一帧的动作参数。

主要组件：
- ActionDiT：动作扩散 Transformer 主模型，基于 DiT 架构，
  以文本编码为条件，对动作序列进行去噪。
- ActionHead：动作预测输出头，将隐藏状态映射回动作空间。

与其他模块的关系：
- 使用与视频 DiT 相同的 DiTBlock 作为基础构建块
- 支持从预训练视频 DiT 中加载主干网络权重（部分迁移学习）
- 支持 MoT（Mixture of Transformers）模式下的行为克隆训练
- 仅使用一维 RoPE（时序维度），无需 3D RoPE
"""

import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from fastwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
)

logger = get_logger(__name__)


class ActionHead(nn.Module):
    """
    动作预测输出头模块。

    与视频 DiT 的 Head 类似，使用 adaLN 调制对隐藏状态进行归一化，
    然后投影到动作空间维度。

    输入特征经过：
    1. LayerNorm（无仿射参数，由调制提供）
    2. adaLN 调制（scale + shift，由时间步条件生成）
    3. 线性投影到动作维度
    """

    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        """
        Args:
            hidden_dim: 隐藏层维度 D
            out_dim: 动作输出维度（动作空间维度）
            eps: LayerNorm epsilon
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        # 可学习的调制参数：2 组（shift 和 scale）
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入隐藏状态, shape [B, S, D]
            t: 时间步嵌入, shape [B, D]

        Returns:
            动作预测输出, shape [B, S, out_dim]
        """
        # 生成 shift 和 scale 调制参数，并添加到序列维度上
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)   # [B, D]
        scale = scale.squeeze(1)   # [B, D]
        # 应用 adaLN 调制后投影到动作空间
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    """
    动作扩散 Transformer 主模型。

    架构与视频 DiT 的前半部分类似，但输入为动作序列而非视频隐空间：
    1. 通过 action_encoder（线性层）将动作向量映射到隐藏空间
    2. 文本条件通过 text_embedding 编码
    3. 时间步通过 time_embedding + time_projection 生成 adaLN 调制参数
    4. 使用一维 RoPE（仅时序维度）进行位置编码
    5. 逐层通过 DiTBlock（自注意力 + 交叉注意力 + FFN）
    6. 通过 ActionHead 映射回动作空间

    特殊功能：
    - backbone_key_set：区分"主干"和"任务特定"参数
    - from_pretrained：从预训练检查点加载权重，支持元数据校验
    """

    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
    ):
        """
        Args:
            hidden_dim: 模型隐藏层维度 D
            action_dim: 动作维度（输入/输出动作空间的维度）
            ffn_dim: FFN 中间层维度
            text_dim: 文本编码维度
            freq_dim: 时间步频率编码维度
            eps: 归一化 epsilon
            num_heads: 注意力头数 H
            attn_head_dim: 每注意力头的维度 Dh
            num_layers: Transformer 层数
            use_gradient_checkpointing: 是否使用梯度检查点
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        # 动作编码器：将原始动作向量映射到隐藏空间
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        # 文本嵌入编码器
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 时间步嵌入编码器
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 时间步投影：hidden_dim -> hidden_dim * 6（adaLN 的 6 组参数）
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        # 堆叠的 DiTBlock
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        # 动作输出头：隐藏状态 -> 动作空间
        self.head = nn.Linear(hidden_dim, action_dim)
        # 预计算一维 RoPE 频率（仅时序维度）
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        """
        从完整的参数键集合中筛选出"主干网络"参数。

        主干网络参数 = 所有参数 - 以 ACTION_BACKBONE_SKIP_PREFIXES 为前缀的参数
        这种方式支持从视频 DiT 迁移学习：加载视频 DiT 的主干权重，
        而动作编码器和输出头保持随机初始化。

        Args:
            keys: 参数键集合

        Returns:
            主干网络参数的键集合（不含 action_encoder. 和 head. 前缀的键）
        """
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        """
        从预训练检查点加载 ActionDiT 模型。

        支持三种加载模式：
        1. 完全加载：从指定路径加载 backbone_state_dict，验证元数据后加载
        2. 跳过加载：skip_dit_load_from_pretrain=True，随机初始化
        3. 无路径提供：随机初始化并返回

        加载时会校验：
        - 元数据（hidden_dim、num_layers 等架构参数）必须匹配
        - backbone_state_dict 中的键必须与模型的 backbone 键完全匹配（无缺失/无多余）
        - 所有张量的 shape 必须一致

        Args:
            action_dit_config: ActionDiT 的配置字典
            action_dit_pretrained_path: 预训练权重路径（支持相对和绝对路径）
            skip_dit_load_from_pretrain: 是否跳过预训练加载
            device: 目标设备
            torch_dtype: 目标精度

        Returns:
            加载权重后的 ActionDiT 实例

        Raises:
            FileNotFoundError: 预训练文件不存在
            ValueError: 配置为空、元数据不匹配、键不匹配、shape 不匹配等
        """
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        # 相对路径基于项目根目录（向上 4 级）
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        # 创建模型实例
        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        # 获取期望的主干网络参数键集合
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        # 加载预训练检查点
        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )

        # 读取加载策略（如果有）
        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        # 校验元数据（架构参数必须匹配）
        meta = payload.get("meta")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        # 加载主干网络状态字典
        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        # 验证键集合完全匹配
        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        # 合并状态字典：主干权重从预训练加载，action_encoder/head 保持随机初始化
        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            # 验证 shape 匹配
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            # 将预训练权重拷贝到目标设备/精度
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        # 严格加载（所有参数必须匹配）
        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Action DiT 前处理：将输入动作序列、文本条件、时间步等转换为 token 序列。

        处理流程：
        1. 验证输入维度
        2. 时间步嵌入编码（正弦编码 -> MLP -> 6组调制参数）
        3. 动作序列通过 action_encoder 映射到隐藏空间
        4. 文本编码投影
        5. 构建 RoPE 频率
        6. 构建注意力掩码

        Args:
            action_tokens: 动作序列, shape [B, T, action_dim]
            timestep: 扩散时间步, shape [B] 或 [1]
            context: 文本编码, shape [B, L, D]
            context_mask: 文本有效掩码, shape [B, L] (1=有效, 0=填充)

        Returns:
            dict 包含:
                - tokens: 动作 token, shape [B, T, D]
                - freqs: RoPE 频率, shape [T, 1, Dh]
                - t: 时间步嵌入, shape [B, D]
                - t_mod: 时间步调制参数, shape [B, 6, D]
                - context: 文本嵌入, shape [B, L, D]
                - context_mask: 扩展后的掩码, shape [B, T, L]
                - meta: 元数据字典
        """
        # --- 输入维度校验 ---
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        # --- batch 一致性校验 ---
        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        # --- 上下文掩码处理 ---
        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        seq_len = action_tokens.shape[1]
        # 校验序列长度不超过 RoPE 预计算长度
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        # --- 时间步嵌入 ---
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        # 投影为 6 组调制参数：shift/scale/gate for SA + shift/scale/gate for MLP
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        # --- 动作序列编码和文本编码 ---
        tokens = self.action_encoder(action_tokens)  # [B, T, action_dim] -> [B, T, D]
        context_emb = self.text_embedding(context)   # [B, L, text_dim] -> [B, L, D]

        # 将 mask 从 [B, L] 扩展到 [B, T, L] 以匹配序列长度
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)

        # RoPE 频率（一维，仅时序）
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        """
        Action DiT 后处理：将 token 映射回动作空间。

        与视频 DiT 不同，Action DiT 的后处理没有 patchify 逆操作，
        直接通过线性层输出动作预测。

        Args:
            tokens: 经过所有 DiTBlock 处理后的 token, shape [B, T, D]
            pre_state: pre_dit 返回的状态字典

        Returns:
            预测的动作, shape [B, T, action_dim]
        """
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        完整的前向传播流程：预处理 -> 逐层 DiTBlock -> 后处理。

        Args:
            action_tokens: 动作序列（含噪声）, shape [B, T, action_dim]
            timestep: 扩散时间步, shape [B] 或 [1]
            context: 文本编码, shape [B, L, D]
            context_mask: 文本掩码, shape [B, L]

        Returns:
            去噪后的动作预测, shape [B, T, action_dim]
        """
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        # 逐层通过 DiTBlock
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state)
