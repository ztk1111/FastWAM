"""
wan_video_dit.py

FastWAM 项目中视频扩散 Transformer（Video DiT）模型的实现。
该模块是 WAN 2.2 视频生成管线中的核心去噪网络，负责将噪声隐空间特征
逐步去噪为高质量的视频隐空间表示。

主要组件：
- WanVideoDiT：视频扩散 Transformer 主模型，支持文本条件、动作条件、
  因果/双向注意力掩码等多种模式。
- DiTBlock：核心 Transformer 块，包含自注意力、交叉注意力和 MLP，
  并带有自适应层归一化（adaLN）调制。
- SelfAttention / CrossAttention：分别实现自注意力和交叉注意力，
  均使用 RoPE 位置编码。
- RMSNorm：均方根层归一化。
- Head：输出头，将隐空间特征映射回像素空间。
- 辅助函数：flash_attention、modulate、sinusoidal_embedding_1d、
  precompute_freqs_cis、rope_apply、create_group_causal_attn_mask 等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Tuple, Optional
from einops import rearrange
from .helpers.gradient import gradient_checkpoint_forward

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor] = None, compatibility_mode=True):
    """
    执行 Flash Attention 计算（兼容模式实现）。

    将 Q/K/V 从 [B, S, H*Dh] 形状重排为多头格式 [B, n, S, Dh]，
    调用 PyTorch 的 scaled_dot_product_attention 进行高效注意力计算，
    再重排回 [B, S, H*Dh]。

    Args:
        q: 查询张量, shape [B, S, H*Dh]
        k: 键张量, shape [B, S, H*Dh]
        v: 值张量, shape [B, S, H*Dh]
        num_heads: 注意力头数
        ctx_mask: 可选的注意力掩码, shape [B, 1, Sq, Sk] 或 [Sq, Sk]
        compatibility_mode: 是否使用兼容模式（必须为 True）

    Returns:
        x: 注意力输出, shape [B, S, H*Dh]
    """
    if compatibility_mode:
        # 将 [B, S, n*d] 重排为 [B, n, S, d] 进行多头注意力计算
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        # PyTorch 原生的缩放点积注意力（含 Flash Attention 优化）
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        # 将输出重排回 [B, S, n*d]
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x
    else:
        raise NotImplementedError("Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True.")



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """
    自适应调制函数，用于 adaLN（自适应层归一化）。

    计算公式: x' = x * (1 + scale) + shift
    其中 scale 和 shift 由时间步条件调制生成。

    Args:
        x: 输入张量, shape [B, S, D]
        shift: 偏移量, shape [B, D] 或广播兼容形状
        scale: 缩放量, shape [B, D] 或广播兼容形状

    Returns:
        调制后的张量, shape 与 x 相同 [B, S, D]
    """
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    """
    一维正弦位置编码。

    使用不同频率的正弦和余弦函数对位置进行编码，
    使得模型能够感知序列中 token 的位置信息。
    编码维度为 dim，各维度的频率按几何级数递减。

    Args:
        dim: 编码维度（需为偶数）
        position: 位置索引, shape [N] 或标量

    Returns:
        位置编码, shape [N, dim]
    """
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    """
    预计算 3D RoPE（旋转位置编码）的频率复数形式。

    视频数据具有时空三维结构（帧、高、宽），本函数为每一维
    独立预计算 RoPE 频率，并按维度分配：剩余维度用于帧维度，
    各 1/3 分别用于高和宽维度。

    Args:
        dim: 每个注意力头的维度（需能被 3 整除的分配方案）
        end: 最大序列长度
        theta: RoPE 基数频率参数

    Returns:
        (f_freqs_cis, h_freqs_cis, w_freqs_cis):
            f_freqs_cis: 帧维度的 RoPE 复数频率, shape [end, dim-2*(dim//3)]
            h_freqs_cis: 高维度的 RoPE 复数频率, shape [end, dim//3]
            w_freqs_cis: 宽维度的 RoPE 复数频率, shape [end, dim//3]
    """
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    """
    预计算一维 RoPE 的频率复数形式。

    按照 RoPE 论文中的公式，为每个位置和每个维度对计算复数旋转因子。
    不同维度使用不同的角频率，从 theta^0 到 theta^(-(dim-2)/dim) 几何衰减。

    Args:
        dim: 维度数（需为偶数）
        end: 最大序列长度
        theta: RoPE 基数（默认 10000.0）

    Returns:
        freqs_cis: 复数形式的 RoPE 频率, shape [end, dim//2], dtype=complex64
    """
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    """
    对输入张量应用 RoPE 旋转位置编码。

    将输入视为复数形式，通过与预计算的复数频率逐元素相乘
    实现旋转操作，使注意力计算能够感知相对位置信息。

    Args:
        x: 输入张量, shape [B, S, H*Dh]
        freqs: RoPE 复数频率, shape [S, Dh//2] (复数)
        num_heads: 注意力头数 H

    Returns:
        应用 RoPE 后的张量, shape [B, S, H*Dh]
    """
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    # 将实数对视为复数：最后两维作为实部和虚部
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    # NPU 设备上需要显式转为 complex64（complex128 不支持）
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    # 复数乘法实现旋转，再转回实数形式
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """
    Creates a group-based attention mask for scaled dot-product attention with two modes:
    'causal' and 'group_diagonal'.

    Parameters:
    - num_temporal_groups (int): The number of temporal groups (e.g., frames in a video sequence).
    - num_query_per_group (int): The number of query tokens per temporal group. (e.g., latent tokens in a frame, H x W).
    - num_key_per_group (int): The number of key tokens per temporal group. (e.g., action tokens per frame).
    - mode (str): The mode of the attention mask. Options are:
        - 'causal': Query tokens can attend to key tokens from the same or previous temporal groups.
        - 'group_diagonal': Query tokens can attend only to key tokens from the same temporal group.

    Returns:
    - attn_mask (torch.Tensor): A boolean tensor of shape (L, S), where:
        - L = num_temporal_groups * num_query_per_group (total number of query tokens)
        - S = num_temporal_groups * num_key_per_group (total number of key tokens)
      The mask indicates where attention is allowed (True) and disallowed (False).

    Example:
    Input:
        num_temporal_groups = 3
        num_query_per_group = 4
        num_key_per_group = 2
    Output:
        Causal Mask Shape: torch.Size([12, 6])
        Group Diagonal Mask Shape: torch.Size([12, 6])
        if mode='causal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True]])

        if mode='group_diagonal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True]])

    """
    assert mode in ["causal", "group_diagonal"], f"Mode {mode} must be 'causal' or 'group_diagonal'"

    # Total number of query and key tokens
    total_num_query_tokens = num_temporal_groups * num_query_per_group  # Total number of query tokens (L)
    total_num_key_tokens = num_temporal_groups * num_key_per_group  # Total number of key tokens (S)

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]

    assert attn_mask.shape == (total_num_query_tokens, total_num_key_tokens), "Attention mask shape mismatch"
    return attn_mask


class RMSNorm(nn.Module):
    """
    RMSNorm（均方根层归一化）模块。

    与标准 LayerNorm 不同，RMSNorm 仅使用均方根统计量进行归一化，
    省略了均值中心化步骤，计算更高效。公式: x / sqrt(mean(x^2) + eps) * weight。

    常用于 Transformer 中的 Q 和 K 归一化，以及注意力计算前的归一化。
    """

    def __init__(self, dim, eps=1e-5):
        """
        Args:
            dim: 归一化维度（特征维度 D）
            eps: 数值稳定小常数，防止除零
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        """
        RMS 归一化计算: x * rsqrt(mean(x^2) + eps)

        Args:
            x: 输入张量, shape [..., D]

        Returns:
            归一化后的张量, shape 与 x 相同 [..., D]
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Args:
            x: 输入张量, shape [..., D]

        Returns:
            缩放后的归一化输出, shape [..., D]
        """
        dtype = x.dtype
        # 在 float32 精度下计算归一化以保持数值稳定性
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    """
    注意力模块封装。

    将多头注意力计算封装为一个模块，内部调用 flash_attention 函数。
    """

    def __init__(self, num_heads):
        """
        Args:
            num_heads: 注意力头数
        """
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, ctx_mask=None):
        """
        Args:
            q: 查询, shape [B, S, H*Dh]
            k: 键, shape [B, S_k, H*Dh]
            v: 值, shape [B, S_k, H*Dh]
            ctx_mask: 可选的注意力掩码

        Returns:
            注意力输出, shape [B, S, H*Dh]
        """
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return x


class SelfAttention(nn.Module):
    """
    自注意力模块，包含 QKV 投影、RMSNorm 和 RoPE 位置编码。

    输入序列中的每个位置都能关注序列中的所有其他位置，
    通过 RoPE 注入相对位置信息。
    """

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        """
        Args:
            hidden_dim: 模型隐藏层维度 D
            attn_head_dim: 每个注意力头的维度 Dh
            num_heads: 注意力头数 H
            eps: RMSNorm 的 epsilon 参数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        # 注意力总维度 = 头数 * 每头维度
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        # QKV 投影: 从隐藏维度 D 映射到注意力维度 H*Dh
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        # 输出投影: 从注意力维度 H*Dh 映射回隐藏维度 D
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        # Q 和 K 在 RoPE 前先进行 RMSNorm
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: 输入张量, shape [B, S, D]
            freqs: RoPE 复数频率, shape [S, Dh//2] (复数)
            self_attn_mask: 可选的注意力掩码, shape [S, S] 或 [B, 1, S, S]

        Returns:
            自注意力输出, shape [B, S, D]
        """
        # Q/K 投影后先做 RMSNorm，再应用 RoPE；V 不做归一化
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        # 对 Q 和 K 应用旋转位置编码 (RoPE)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        # 执行 Flash Attention
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    """
    交叉注意力模块，用于将条件信息（如文本编码）注入到主序列中。

    Query 来自主序列（如视频 token），Key/Value 来自条件序列（如文本 token）。
    """

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,):
        """
        Args:
            hidden_dim: 模型隐藏层维度 D
            attn_head_dim: 每个注意力头的维度 Dh
            num_heads: 注意力头数 H
            eps: RMSNorm 的 epsilon 参数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        # Q 来自主序列，K/V 来自条件序列
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        # 仅对 Q 和 K 做 RMSNorm
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: 主序列（Query 来源）, shape [B, S, D]
            ctx: 条件序列（Key/Value 来源）, shape [B, L, D]
            ctx_mask: 条件序列的注意力掩码, shape [B, L] 或 [B, 1, S, L]

        Returns:
            交叉注意力输出, shape [B, S, D]
        """
        # Q 来自主序列 x，K/V 来自条件序列 ctx
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        # 交叉注意力不使用 RoPE（位置信息由条件序列本身携带）
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class GateModule(nn.Module):
    """
    门控残差连接模块。

    实现带可学习门控的残差连接: x + gate * residual
    门控值由时间步调制生成，控制残差贡献的强度。
    """

    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        """
        Args:
            x: 主路径输入, shape [B, S, D]
            gate: 门控值, shape [B, D] 或广播兼容形状
            residual: 残差路径输出, shape [B, S, D]

        Returns:
            门控残差输出, shape [B, S, D]
        """
        return x + gate * residual


class DiTBlock(nn.Module):
    """
    DiT（Diffusion Transformer）核心块。

    包含三个子模块的顺序处理：
    1. 自注意力（Self-Attention）：带 adaLN 调制和 RoPE
    2. 交叉注意力（Cross-Attention）：注入文本条件信息
    3. 前馈网络（MLP/FFN）：带 adaLN 调制

    每个子模块都使用自适应层归一化（adaLN），其缩放/偏移/门控参数
    由时间步调制条件动态生成。
    """

    def __init__(self,  hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        """
        Args:
            hidden_dim: 模型隐藏层维度 D
            attn_head_dim: 每个注意力头的维度 Dh
            num_heads: 注意力头数 H
            ffn_dim: 前馈网络中间层维度
            eps: LayerNorm 的 epsilon 参数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # 自注意力模块（含 RoPE）
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        # 交叉注意力模块（用于文本条件注入）
        self.cross_attn = CrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps)
        # 三个 LayerNorm（elementwise_affine=False 表示由 adaLN 提供缩放/偏移）
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        # 前馈网络: D -> FFN_DIM -> D，中间使用 GELU 激活
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        # 可学习的调制参数: 6 组（shift/scale/gate 各 2 组，分别用于 SA 和 MLP）
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        # 门控残差模块
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: 输入 token 序列, shape [B, S, D]
            context: 文本条件编码, shape [B, L, D]
            t_mod: 时间步调制条件, shape [B, 6, D] 或 [B, 6, 1, D]（逐 token 调制）
            freqs: RoPE 频率, shape [S, 1, Dh//2]
            context_mask: 条件掩码, shape [B, L] 或 [B, 1, S, L]
            self_attn_mask: 自注意力掩码, shape [S, S]

        Returns:
            处理后 token 序列, shape [B, S, D]
        """
        # 如果 context_mask 是 3D，在第 1 维插入头维度
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)  # (B, 1, seq_len, context_len), 1 for heads
        # t_mod 的第 2 维是否包含序列维度（逐 token 调制 vs 全局调制）
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        # 将 6 组调制参数拆分为: SA 的 shift/scale/gate 和 MLP 的 shift/scale/gate
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            # 逐 token 调制时，压缩掉 sequence 维度
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        # 1. 自注意力前做 adaLN 调制，然后经过门控残差连接
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        # 2. 交叉注意力（注入文本条件），恒等残差连接
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        # 3. MLP 前做 adaLN 调制，然后经过门控残差连接
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    """
    多层感知机映射模块。

    将输入从 in_dim 维度映射到 out_dim 维度，包含 LayerNorm 和 GELU 激活。
    可选择性地添加可学习的位置编码。
    """

    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        """
        Args:
            in_dim: 输入维度
            out_dim: 输出维度
            has_pos_emb: 是否包含可学习位置编码
        """
        super().__init__()
        # 序列化处理: LayerNorm -> Linear -> GELU -> Linear -> LayerNorm
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            # 可学习位置编码，支持最多 514 个 token，维度 1280
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        """
        Args:
            x: 输入张量, shape [B, S, in_dim]

        Returns:
            输出张量, shape [B, S, out_dim]
        """
        if self.has_pos_emb:
            # 添加可学习位置编码
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    """
    DiT 输出头模块。

    将去噪后的隐空间 token 映射回像素空间。
    包含输出投影、LayerNorm 和 adaLN 调制。
    输出维度 = out_dim * patch_size[0] * patch_size[1] * patch_size[2]，
    以便重组为原始图像/视频形状。
    """

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        """
        Args:
            dim: 隐藏层维度 D
            out_dim: 输出通道数（如 VAE 隐空间通道数）
            patch_size: (temporal_patch, height_patch, width_patch)
            eps: LayerNorm epsilon
        """
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        # 输出投影: D -> out_dim * prod(patch_size)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        # 调制参数：2 组（shift 和 scale）
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        """
        Args:
            x: 输入 token, shape [B, S, D]
            t_mod: 时间步调制条件, shape [B, D] 或 [B, T, D]（逐帧调制）

        Returns:
            输出 token, shape [B, S, out_dim * prod(patch_size)]
        """
        if len(t_mod.shape) == 3:
            # 逐帧调制模式: t_mod shape [B, T, D]
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            # 全局调制模式: t_mod shape [B, D]
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanVideoDiT(torch.nn.Module):
    """
    WanVideoDiT 主模型：用于视频生成的扩散 Transformer。

    整体流程：
    1. pre_dit: 输入预处理
       - 将 5D 视频隐空间 [B, C, T, H, W] 通过 PatchEmbed 转换为 3D token [B, S, D]
       - 对文本条件进行嵌入编码
       - 生成时间步调制条件（逐帧或全局）
       - 构建 RoPE 频率
       - 如果启用动作条件，对动作序列进行编码并构建注意力掩码
    2. 逐层通过 DiTBlock（自注意力 + 交叉注意力 + FFN）
    3. post_dit: 通过 Head 将 token 映射回像素空间并还原为 5D 形状

    关键特性：
    - 支持双向、逐帧因果、首帧因果等多种注意力掩码模式
    - 支持动作条件生成（action_conditioned）
    - 支持梯度检查点以节省显存
    - 3D RoPE 位置编码（帧、高、宽三个维度）
    - adaLN 时间步调制
    """

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode = "causal",
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
    ):
        """
        Args:
            hidden_dim: 模型隐藏层维度 D
            in_dim: 输入通道数（VAE 隐空间通道数）
            ffn_dim: FFN 中间层维度
            out_dim: 输出通道数
            text_dim: 文本编码维度
            freq_dim: 时间步频率编码维度
            eps: 归一化 epsilon
            patch_size: (temporal_patch, height_patch, width_patch)
            num_heads: 注意力头数 H
            attn_head_dim: 每注意力头的维度 Dh
            num_layers: Transformer 层数
            has_image_input: 是否有图像输入（当前不支持）
            has_image_pos_emb: 是否使用图像位置编码
            has_ref_conv: 是否使用参考图卷积
            add_control_adapter: 是否添加控制适配器
            in_dim_control_adapter: 控制适配器输入维度
            seperated_timestep: 是否使用逐帧分离的时间步
            require_vae_embedding: 是否需要 VAE 嵌入（不支持）
            require_clip_embedding: 是否需要 CLIP 嵌入（不支持）
            fuse_vae_embedding_in_latents: 是否在隐空间中融合 VAE 嵌入
            action_conditioned: 是否启用动作条件生成
            action_dim: 动作维度
            action_group_causal_mask_mode: 动作分组因果掩码模式
            video_attention_mask_mode: 视频自注意力掩码模式
            use_gradient_checkpointing: 是否使用梯度检查点
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}"
            )

        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        assert has_image_input == False
        assert require_clip_embedding == False
        assert require_vae_embedding == False and fuse_vae_embedding_in_latents == True, "Only support fusing vae embedding in latents"

        # Patch Embed: 将 5D 视频隐空间分解为非重叠 patch token
        self.patch_embedding = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        # 文本嵌入: text_dim -> hidden_dim
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 时间步嵌入: freq_dim -> hidden_dim
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 时间步投影: hidden_dim -> hidden_dim * 6（用于 adaLN 的 6 组参数）
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        # 堆叠多个 DiTBlock
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        # 输出头
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        # 预计算 3D RoPE 频率
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        # 动作条件模块：将动作向量编码到隐藏空间
        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode

        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info("Using gradient checkpointing for DiT blocks. This will save memory but use more computation.")


    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        """
        将 5D 视频张量通过 3D 卷积分解为 patch token。

        Args:
            x: 输入视频隐空间, shape [B, C, T, H, W]
            control_camera_latents_input: 可选的相机控制隐空间输入

        Returns:
            x: patch token, shape [B, D, T//pt, H//ph, W//pw]
        """
        x = self.patch_embedding(x)
        # 如果启用了控制适配器，融合相机控制信号
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        """
        将 token 序列还原为原始 5D 视频形状（patchify 的逆操作）。

        Args:
            x: token 序列, shape [B, T'*H'*W', out_dim*pt*ph*pw]
            grid_size: (T', H', W') 即 patch 后的网格尺寸

        Returns:
            还原后的视频张量, shape [B, out_dim, T, H, W]
        """
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        验证前向传播输入的有效性，包括维度检查、形状匹配等。

        检查项：
        - x 必须是 5D [B, C, T, H, W]
        - context 必须是 3D [B, L, D]
        - timestep 必须是 1D [B] 或 [1]
        - batch size 一致性
        - 动作条件模式下 action 的形状和有效性

        Args:
            x: 输入视频隐空间, shape [B, C, T, H, W]
            timestep: 时间步, shape [B] 或 [1]
            context: 文本编码, shape [B, L, D]
            context_mask: 文本掩码, shape [B, L]
            action: 动作序列（可选）, shape [B, action_horizon, action_dim]

        Returns:
            (x, timestep, context_mask): 处理/扩展后的张量
        """
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            # 允许单帧文本推理模式（无需动作输入）
            allow_text_only_single_frame = (num_latent_frames == 1 and action is None)
            if not allow_text_only_single_frame:
                assert action is not None, "Action input is required for action-conditioned model."
                if action.ndim != 3:
                    raise ValueError(f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}")
                if action.shape[2] != self.action_dim:
                    raise ValueError(f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}")
                if num_latent_frames <= 1:
                    raise ValueError(f"video length must be > 1 for action-conditioned model, got {num_latent_frames}")
                # 动作序列长度必须能被 (帧数-1) 整除，因为每帧对应一组动作
                if action.shape[1] % (num_latent_frames - 1) != 0:
                    raise ValueError(
                        f"action horizon must be divisible by (num_latent_frames - 1), got action_horizon={action.shape[1]}"
                    )
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}")

        batch_size = x.shape[0]
        # 如果 batch size 不匹配且在推理模式，自动扩展视频 batch
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(
                    f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}."
                )

        # 时间步长度可以是 1（共享）或 batch_size
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            assert not self.training, "During training, timestep length must match batch_size."
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        构建视频自注意力掩码。

        支持三种模式：
        - bidirectional: 全双向注意力（所有帧互相可见）
        - per_frame_causal: 逐帧因果注意力（帧间因果，帧内双向）
        - first_frame_causal: 首帧因果（仅第一帧关注所有帧，后续帧不能回头看第一帧）

        Args:
            video_seq_len: 视频 token 序列长度 S
            video_tokens_per_frame: 每帧的 token 数
            device: 目标设备

        Returns:
            attention_mask: 布尔注意力掩码, shape [S, S]
        """
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            # 全双向：所有位置互相可见
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            # 逐帧因果：帧间下三角掩码，每帧内的所有 token 共享相同的帧级因果模式
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            # 构建帧级因果掩码（下三角）
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            # 扩展到 token 级别：每帧的 token 共享相同的帧级注意力模式
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            # 首帧因果：第一帧能看所有帧，后续帧不能看第一帧
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        DiT 前处理：将输入视频、文本条件、时间步等转换为 token 序列。

        处理流程：
        1. 验证输入维度
        2. 对视频进行 patchify（3D Conv -> [B, D, T', H', W']）
        3. 生成时间步调制条件 t_mod
        4. 编码文本条件
        5. 如果启用动作条件，编码动作序列并构建注意力掩码
        6. 构建 3D RoPE 频率
        7. 重排 token 为 [B, S, D] 格式

        Args:
            x: 输入视频隐空间, shape [B, C, T, H, W]
            timestep: 扩散时间步, shape [B] 或 [1]
            context: 文本编码, shape [B, L, D]
            context_mask: 文本有效掩码, shape [B, L] (1=有效, 0=填充)
            action: 动作序列（可选）, shape [B, action_horizon, action_dim]
            fuse_vae_embedding_in_latents: 是否在隐空间中融合 VAE 嵌入
            control_camera_latents_input: 相机控制隐空间输入（可选）

        Returns:
            dict 包含:
                - tokens: 视频 token, shape [B, S, D]
                - freqs: RoPE 频率, shape [S, 1, Dh]
                - t: 时间步嵌入, shape [B, S, D] 或 [B, D]
                - t_mod: 时间步调制参数, shape [B, T', 6, D] 或 [B, 6, D]
                - context: 文本/条件嵌入, shape [B, L+action_len, D]
                - context_mask: 条件掩码, shape [B, S, L+action_len]
                - meta: 元数据字典
        """
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        # 检查空间维度是否能被 patch 大小整除
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        # 每帧的 token 数 = H' * W'
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            if not hasattr(self, "patch_size") or len(self.patch_size) < 3:
                raise ValueError(f"Invalid dit.patch_size: {getattr(self, 'patch_size', None)}")

            # 为每个帧生成独立的时间步（第一帧时间步设为 0，即参考帧无条件）
            token_timesteps = torch.ones(
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            token_timesteps[:, 0, :] = 0  # 第一帧为参考帧，时间步为 0
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            # 将时间嵌入投影为 6 组调制参数（用于 adaLN）
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            raise NotImplementedError("Only support seperated_timestep with fuse_vae_embedding_in_latents for now.")
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        # 视频 patchify: [B, C, T, H, W] -> [B, D, T', H', W']
        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]  # patch 后的时空网格尺寸

        # 文本编码投影
        context = self.text_embedding(context)  # (B, L, dim)
        context_len = context.shape[1]
        # 如果启用动作条件且提供了动作，拼接动作嵌入到条件序列中
        if self.action_conditioned and action is not None:
            action_len = action.shape[1]
            action_emb = self.action_embedding(action)  # (B, action_len, dim)
            # 为动作序列添加正弦位置编码
            action_pos_embed = sinusoidal_embedding_1d(self.hidden_dim,
                torch.arange(action_len, device=action_emb.device))  # (action_len, dim)
            action_emb = action_emb + action_pos_embed.unsqueeze(0)  # (B, action_len, dim)
            # 将动作嵌入拼接到文本嵌入后面
            context = torch.cat([context, action_emb], dim=1)  # (B, context_len + action_len, dim)

            # 构建动作条件注意力掩码
            num_temporal_groups = f - 1  # 第一帧不关注动作（参考帧）
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned context mask requires at least 2 latent frames when `action` is provided."
                )
            assert action_emb.shape[1] % num_temporal_groups == 0, \
                f"Action embedding length {action_emb.shape[1]} must be divisible by number of temporal groups {num_temporal_groups}"
            # Each latent frame (from the 2nd one) attends to the corresponding group of action tokens
            # 每帧（从第二帧开始）关注对应分组的动作 token
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device)  # ((f-1)*tokens_per_frame, action_len)

            seq_len = f * h * w  # query 序列长度
            # 最终的上下文掩码: [B, seq_len, L + action_len]
            final_context_mask = torch.zeros((batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device)
            # 所有帧都能关注文本 token
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq_len, L)
            # 第二帧开始的帧按分组关注动作 token
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(0).expand(batch_size, -1, -1)  # (B, seq_len, action_len)
            context_mask = final_context_mask
        elif self.action_conditioned and action is None:
            # 动作条件模型但无动作输入：仅单帧文本推理模式
            if f != 1:
                raise ValueError(
                    "Action-conditioned model requires `action` unless running single-frame text-only mode with num_latent_frames=1."
                )
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1)  # (B, S, L)
        else:
            # 非动作条件模式：直接将 mask 扩展到序列长度
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1)  # (B, S, L)

        # 重排为 token 序列: [B, D, F, H, W] -> [B, S=F*H*W, D]
        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        # 构建 3D RoPE 频率（帧、高、宽三轴频率拼合）
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x_tokens.device)

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        """
        DiT 后处理：将 token 通过 Head 映射回像素空间。

        Args:
            x_tokens: 去噪后的 token, shape [B, S, D]
            pre_state: pre_dit 返回的状态字典，包含 grid_size 等元数据

        Returns:
            x: 重建的视频张量, shape [B, out_dim, T, H, W]
        """
        f, h, w = pre_state["meta"]["grid_size"]
        # 通过 Head 将 D 维映射到 out_dim * prod(patch_size) 维
        x = self.head(x_tokens, pre_state["t"])
        # 还原为 5D 视频形状
        x = self.unpatchify(x, (f, h, w))
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        """
        完整的前向传播流程：预处理 -> 逐层 DiTBlock -> 后处理。

        Args:
            x: 输入视频隐空间, shape [B, C, T, H, W]
            timestep: 扩散时间步, shape [B] 或 [1]
            context: 文本编码, shape [B, L, D]
            context_mask: 文本掩码, shape [B, L]
            action: 动作序列（可选）, shape [B, action_horizon, action_dim]
            fuse_vae_embedding_in_latents: 是否融合 VAE 嵌入

        Returns:
            去噪后的视频张量, shape [B, out_dim, T, H, W]
        """
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        # 构建视频自注意力掩码（双向模式不需要掩码，可加速计算）
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None  # special rule for faster speed

        # 逐层通过所有 DiTBlock
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                # 使用梯度检查点节省显存（以额外计算换显存）
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask
                )
            else:
                x_tokens = block(x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask)

        return self.post_dit(x_tokens, pre_state)
