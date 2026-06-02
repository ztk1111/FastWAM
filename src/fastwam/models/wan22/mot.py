"""
mot.py

FastWAM 项目中 MoT（Mixture of Transformers，混合变换器）模块的实现。

MoT 的核心思想是在同一模型中混合多个专家（Expert），每个专家拥有独立的
DiTBlock 权重，但在每层的注意力计算时进行"混合注意力（Mixed Attention）"：
所有专家的 Q、K、V 被拼接后在一个统一的注意力掩码下进行计算，
然后将注意力输出按专家分开，每个专家独立进行后续的 MLP 处理。

关键特性：
1. 混合注意力（Mixed Attention）：视频和动作专家在每层的自注意力阶段
   共享一个联合注意力矩阵，实现跨模态信息交换。
2. KV 缓存机制：视频专家可以预填充 KV 缓存，在后续的动作去噪过程中
   重复使用，避免重复计算视频分支。
3. 梯度检查点：支持对混合注意力和后处理块进行梯度检查点，节省显存。
4. 每个专家的 MLP 和交叉注意力仍然是独立的。

主要使用场景：
- 行为克隆（Behavior Cloning）：同时处理视频观察和动作预测
- 推理时通过 prefill_video_cache + forward_action_with_video_cache
  实现高效的视频条件动作生成
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention, modulate, rope_apply
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    """
    混合变换器（Mixture of Transformers）主模块。

    同时管理多个"专家"（如 video 和 action），每个专家拥有独立的
    DiTBlock 权重。在每层的前向传播中：
    1. 对每个专家独立计算 Q/K/V（通过各自的 SelfAttention 投影）
    2. 将所有专家的 Q/K/V 拼接，在统一掩码下执行混合注意力
    3. 将注意力输出分割回各专家
    4. 各专家独立进行交叉注意力和 MLP 处理

    支持两种执行模式：
    - 联合前向（forward）：所有专家同时计算，用于训练。
    - 视频预填充 + 动作推理（prefill_video_cache + forward_action_with_video_cache）：
      先对视频分支做一次完整计算并缓存每层的 K/V，
      然后动作分支在每层复用视频的 K/V，避免重复计算。用于推理加速。
    """

    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        """
        Args:
            mixtures: 专家名字到专家模块的映射字典。
                      必须至少包含 'video' 和 'action' 两个专家。
                      每个专家模块需包含：
                        - blocks: DiTBlock 列表
                        - num_heads: 注意力头数
                        - attn_head_dim: 每头维度
            mot_checkpoint_mixed_attn: 是否对混合注意力使用梯度检查点
        """
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())  # 保持专家顺序一致
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        # 从第一个专家获取架构参数并验证所有专家的一致性
        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )

        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        """
        将时间步调制条件拆分为 6 个调制参数。

        从 block.modulation（可学习基础调制）和 t_mod（条件相关调制）
        之和中分离出 SA 和 MLP 各自的 shift/scale/gate。

        Args:
            block: DiTBlock 实例，包含 .modulation 参数
            t_mod: 时间步调制, shape [B, 6, D] 或 [B, 6, 1, D]（逐 token）

        Returns:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp):
                每组 shape [B, D] 或 [B, 1, D]（逐 token 调制时）
        """
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        # 基础调制 + 条件调制
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            # 逐 token 调制时，压缩掉额外的维度
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行混合注意力计算。

        将所有专家的 Q/K/V 拼接后进行统一的 Flash Attention 计算，
        使得不同专家之间能够在注意力层进行信息交换。
        支持梯度检查点以节省训练显存。

        Args:
            q_cat: 拼接后的查询, shape [B, S_total, H*Dh]
            k_cat: 拼接后的键, shape [B, S_total, H*Dh]
            v_cat: 拼接后的值, shape [B, S_total, H*Dh]
            attention_mask: 联合注意力掩码, shape [S_total, S_total]

        Returns:
            混合注意力输出, shape [B, S_total, H*Dh]
        """
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            # 梯度检查点：在前向时丢弃中间激活，反向时重新计算
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """
        对单个专家执行注意力后的 MLP 和交叉注意力处理。

        流程：
        1. 自注意力残差连接（门控）：x = x + gate_msa * SA_O(mixed_attn_out)
        2. 交叉注意力（如果提供了上下文条件）
        3. MLP 残差连接（门控）：x = x + gate_mlp * FFN(adaLN(x))

        Args:
            block: 该专家的 DiTBlock
            residual_x: 注意力前的原始输入（用于残差）, shape [B, S, D]
            mixed_attn_out: 混合注意力输出（经过 SA 的 o 投影后）, shape [B, S, Dh*H]
            gate_msa: SA 门控, shape [B, D] 或广播兼容形状
            shift_mlp: MLP adaLN 偏移, shape [B, D]
            scale_mlp: MLP adaLN 缩放, shape [B, D]
            gate_mlp: MLP 门控, shape [B, D]
            context_payload: 可选的条件上下文，包含：
                - context: 条件编码, shape [B, L, D]
                - mask: 注意力掩码, shape [B, S, L] 或 [B, 1, S, L]

        Returns:
            该专家更新后的 token, shape [B, S, D]
        """
        # 1. 自注意力残差：block.self_attn.o 从注意力空间投影回隐藏空间
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        # 2. 带条件的交叉注意力（如果需要）
        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    # 插入头维度 [B, S, L] -> [B, 1, S, L]
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        # 3. MLP 处理 + 门控残差
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """
        为单个专家构建注意力输入所需的 Q/K/V 及后续处理所需的状态。

        在 DiTBlock 内部，此函数执行：
        1. 拆分 adaLN 调制参数
        2. 对输入进行调制和 LayerNorm
        3. 通过 Q/K 投影、RMSNorm、RoPE 得到 q 和 k
        4. 通过 V 投影得到 v
        5. 保存残差路径所需的原始 x 和调制参数

        Args:
            expert: 该专家模块（用于读取 use_gradient_checkpointing 配置）
            block: 该专家的当前层 DiTBlock
            x: 当前专家 token, shape [B, S, D]
            freqs: RoPE 频率, shape [S, 1, rope_dim]
            t_mod: 时间步调制, shape [B, 6, D] 或 [B, 6, 1, D]（逐 token）

        Returns:
            q: 查询（已归一化和 RoPE）, shape [B, S, H*Dh]
            k: 键（已归一化和 RoPE）, shape [B, S, H*Dh]
            v: 值, shape [B, S, H*Dh]
            residual_x: 原始输入 x（用于后处理残差）, shape [B, S, D]
            gate_msa: SA 门控, shape [B, D]
            shift_mlp: MLP 偏移, shape [B, D]
            scale_mlp: MLP 缩放, shape [B, D]
            gate_mlp: MLP 门控, shape [B, D]
            use_gradient_checkpointing: 该专家是否启用梯度检查点
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        # 对输入进行 adaLN 调制作为自注意力输入
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        # Q 投影 + RMSNorm
        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        # K 投影 + RMSNorm
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        # V 投影（不做归一化）
        v = block.self_attn.v(attn_input)

        # 对 Q 和 K 应用 RoPE 旋转位置编码
        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,             # 残差输入
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """
        执行注意力后处理，可选择使用梯度检查点。

        包装 _apply_expert_post_block 函数，使其支持
        torch.utils.checkpoint.checkpoint 的调用签名。

        Args:
            block: 该专家的 DiTBlock
            residual_x: 残差输入, shape [B, S, D]
            gate_msa: SA 门控, shape [B, D]
            shift_mlp: MLP 偏移, shape [B, D]
            scale_mlp: MLP 缩放, shape [B, D]
            gate_mlp: MLP 门控, shape [B, D]
            use_gradient_checkpointing: 是否使用梯度检查点
            mixed_slice: 分配给该专家的混合注意力输出, shape [B, S, H*Dh]
            context_payload: 可选的条件上下文字典

        Returns:
            该专家更新后的 token, shape [B, S, D]
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """
        预填充视频分支并缓存每层的 K/V。

        用于推理加速：对视频分支执行一次完整前向传播，
        将每层的 Key 和 Value 缓存起来，后续动作分支可以直接复用。

        执行流程：
        1. 逐层通过视频专家的 DiTBlock
        2. 每层计算 Q/K/V 后执行自注意力（仅视频掩码）
        3. 保存每层的 K/V 到缓存列表
        4. 同时更新视频 token 供下一层使用

        Args:
            video_tokens: 视频 token（第 0 层输入）, shape [B, Sv, D]
            video_freqs: 视频 RoPE 频率, shape [Sv, 1, rope_dim]
            video_t_mod: 视频时间步调制
            video_context_payload: 视频条件上下文（可选）
                - context: 文本编码, shape [B, L, D]
                - mask: 掩码, shape [B, Sv, L] 或 [B, 1, Sv, L]
            video_attention_mask: 视频自注意力掩码, shape [Sv, Sv]

        Returns:
            kv_cache: 每层的 KV 缓存列表，长度等于 num_layers
                每个元素为 dict: {"k": tensor[B, Sv, H*Dh], "v": tensor[B, Sv, H*Dh]}
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            # 从当前层输入构建视频的 Q/K/V
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            # 视频预填充阶段仅使用视频自注意力掩码（无动作 token）
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            # 更新视频 token 供下一层使用，并保存当前层的 K/V
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """
        使用缓存的视频 KV 运行动作分支（推理加速模式）。

        在此模式下，视频分支已经通过 prefill_video_cache 计算完成并缓存了每层的 K/V。
        动作分支每层只需计算自己的 Q/K/V，然后与缓存的视频 K/V 拼接后进行混合注意力。

        相比联合前向（forward），此模式避免了视频分支的重复计算，提高推理速度。

        Args:
            action_tokens: 动作 token（第 0 层输入）, shape [B, Sa, D]
            action_freqs: 动作 RoPE 频率, shape [Sa, 1, rope_dim]
            action_t_mod: 动作时间步调制
            action_context_payload: 动作条件上下文（可选）
            video_kv_cache: prefill_video_cache 返回的缓存列表
            attention_mask: 联合注意力掩码, shape [Sv+Sa, Sv+Sa]
            video_seq_len: 视频 token 数 Sv（在联合序列中作为前缀）

        Returns:
            更新后的动作 token（经过所有层）, shape [B, Sa, D]
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        # 从联合掩码中提取动作对应的行（动作 query 可以看到视频前缀和自身）
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            # 动作的 Q/K/V 依赖于当前时间步，必须每步重新计算
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            # 混合注意力：动作 query 同时关注缓存的视频 K/V 和自身的 K/V
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        """
        联合前向传播（训练模式）：所有专家同时计算。

        流程（每层）：
        1. 对每个专家：计算 Q/K/V 和残差状态
        2. 拼接所有专家的 Q/K/V
        3. 在联合注意力掩码下执行混合注意力
        4. 将注意力输出按专家分割
        5. 每个专家独立执行后处理（交叉注意力 + MLP）
        6. 更新每个专家的 token 供下一层使用

        Args:
            embeds_all: 各专家的输入 token 字典
                格式: {expert_name: tensor[B, S_k, D]}
            attention_mask: 联合注意力掩码, shape [S_total, S_total]
            freqs_all: 各专家的 RoPE 频率字典
                格式: {expert_name: tensor[S_k, 1, rope_dim]}
            context_all: 各专家的条件上下文字典（可选）
                格式: {expert_name: dict 或 None}
                dict 包含:
                    - context: 条件编码, shape [B, L, D]
                    - mask: 掩码, shape [B, S_k, L] 或 [B, 1, S_k, L]
            t_mod_all: 各专家的时间步调制字典
                格式: {expert_name: tensor[B, 6, D]}

        Returns:
            tokens_all: 各专家更新后的 token 字典
                格式: {expert_name: tensor[B, S_k, D]}
        """
        # 验证所有必需的专家输入都存在
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        # 深拷贝 token 字典，避免修改原始输入
        tokens_all = {k: v for k, v in embeds_all.items()}

        # 逐层处理
        for layer_idx in range(self.num_layers):
            q_chunks = []     # 各专家的 Q 拼接列表
            k_chunks = []     # 各专家的 K 拼接列表
            v_chunks = []     # 各专家的 V 拼接列表
            cached = {}       # 各专家的后处理状态缓存
            seq_lens = []     # 各专家的序列长度

            # 1. 对每个专家构建注意力输入
            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                # 收集 Q/K/V 和序列长度
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                # 缓存后处理所需的状态
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 2. 拼接所有专家的 Q/K/V 进行混合注意力
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            # 3. 执行混合注意力
            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)

            # 4. 分割注意力输出并分别进行后处理
            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]  # 分配给该专家的输出
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)  # 该专家的条件上下文

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
