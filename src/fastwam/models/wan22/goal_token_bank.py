"""
Goal Token Bank encoder module.

Provides a lightweight goal-space encoder that maps T5 text embeddings to a small set
of goal tokens via learned queries and cross-attention. Weights are loaded from an
alignment checkpoint (experiments/goal_alignment).

Only the text encoding path is used at runtime; image-path weights are retained
solely for alignment-checkpoint compatibility (`strict=False` loading).
"""

import torch
import torch.nn as nn


class GoalTokenBank(nn.Module):
    """Encodes T5 text context into goal tokens [B, M, goal_dim].

    Architecture mirrors ``GoalProjector`` from experiments/goal_alignment/common.py
    so that alignment checkpoints load with minimal key mismatches.
    """

    def __init__(
        self,
        text_dim: int = 4096,
        goal_dim: int = 512,
        num_goal_tokens: int = 4,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_goal_tokens = int(num_goal_tokens)
        self.goal_dim = int(goal_dim)

        # -- text encoding path (used at runtime) --
        self.text_queries = nn.Parameter(torch.randn(self.num_goal_tokens, self.goal_dim) * 0.02)
        self.text_input = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, self.goal_dim),
        )
        self.text_attn = nn.MultiheadAttention(
            embed_dim=self.goal_dim,
            num_heads=int(num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.text_ffn = nn.Sequential(
            nn.LayerNorm(self.goal_dim),
            nn.Linear(self.goal_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.goal_dim),
        )
        self.text_output_norm = nn.LayerNorm(self.goal_dim)

        # -- image encoding path (kept for alignment-ckpt compatibility, unused at runtime) --
        self.image_queries = nn.Parameter(torch.randn(self.num_goal_tokens, self.goal_dim) * 0.02)
        self.image_input = nn.Sequential(
            nn.LayerNorm(48),
            nn.Linear(48, self.goal_dim),
        )
        self.image_attn = nn.MultiheadAttention(
            embed_dim=self.goal_dim,
            num_heads=int(num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.image_ffn = nn.Sequential(
            nn.LayerNorm(self.goal_dim),
            nn.Linear(self.goal_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.goal_dim),
        )
        self.image_output_norm = nn.LayerNorm(self.goal_dim)

    def encode_text(self, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        """Encode T5 text context into goal tokens.

        Args:
            context:  [B, L, D]  T5 text embeddings.
            context_mask:  [B, L]  bool (True = valid token).

        Returns:
            [B, M, goal_dim] goal tokens.
        """
        batch_size = int(context.shape[0])
        queries = self.text_queries.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = self.text_input(context)
        key_padding_mask = None if context_mask is None else ~context_mask.to(device=context.device, dtype=torch.bool)
        attended, _ = self.text_attn(
            query=queries,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.text_output_norm(attended + self.text_ffn(attended))
