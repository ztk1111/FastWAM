from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionVAEOutput:
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


class ActionChunkVAE(nn.Module):
    """VAE that compresses one action chunk into a small set of latent tokens."""

    def __init__(
        self,
        action_dim: int,
        chunk_len: int = 32,
        latent_tokens: int = 3,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if chunk_len <= 0:
            raise ValueError(f"`chunk_len` must be positive, got {chunk_len}.")
        if latent_tokens <= 0:
            raise ValueError(f"`latent_tokens` must be positive, got {latent_tokens}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"`hidden_dim` must be divisible by `num_heads`, got {hidden_dim}/{num_heads}.")

        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.latent_tokens = int(latent_tokens)
        self.latent_dim = int(latent_dim)

        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.action_pos = nn.Parameter(torch.zeros(chunk_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.latent_queries = nn.Parameter(torch.randn(latent_tokens, hidden_dim) / math.sqrt(hidden_dim))
        self.latent_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.decode_queries = nn.Parameter(torch.randn(chunk_len, hidden_dim) / math.sqrt(hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.action_pos, std=0.02)
        nn.init.normal_(self.decode_queries, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, action: torch.Tensor, action_is_pad: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if action.ndim != 3:
            raise ValueError(f"`action` must be [B,T,D], got {tuple(action.shape)}.")
        if action.shape[1] != self.chunk_len or action.shape[2] != self.action_dim:
            raise ValueError(
                f"`action` must have shape [B,{self.chunk_len},{self.action_dim}], got {tuple(action.shape)}."
            )

        x = self.action_proj(action) + self.action_pos.unsqueeze(0)
        key_padding_mask = action_is_pad.bool() if action_is_pad is not None else None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        queries = self.latent_queries.unsqueeze(0).expand(action.shape[0], -1, -1)
        latent_h, _ = self.latent_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.to_mu(latent_h), self.to_logvar(latent_h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or z.shape[1] != self.latent_tokens or z.shape[2] != self.latent_dim:
            raise ValueError(
                f"`z` must have shape [B,{self.latent_tokens},{self.latent_dim}], got {tuple(z.shape)}."
            )
        memory = self.latent_proj(z)
        tgt = self.decode_queries.unsqueeze(0).expand(z.shape[0], -1, -1)
        decoded = self.decoder(tgt=tgt, memory=memory)
        return self.out(decoded)

    def forward(self, action: torch.Tensor, action_is_pad: torch.Tensor | None = None) -> ActionVAEOutput:
        mu, logvar = self.encode(action, action_is_pad=action_is_pad)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return ActionVAEOutput(reconstruction=reconstruction, mu=mu, logvar=logvar, z=z)


def action_vae_loss(
    output: ActionVAEOutput,
    target: torch.Tensor,
    action_is_pad: torch.Tensor | None = None,
    beta: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon_error = F.mse_loss(output.reconstruction, target, reduction="none").mean(dim=-1)
    if action_is_pad is not None:
        valid = (~action_is_pad.bool()).to(dtype=recon_error.dtype)
        recon_loss = (recon_error * valid).sum() / valid.sum().clamp_min(1.0)
    else:
        recon_loss = recon_error.mean()

    kl = -0.5 * (1.0 + output.logvar - output.mu.pow(2) - output.logvar.exp())
    kl_loss = kl.mean()
    loss = recon_loss + float(beta) * kl_loss
    metrics = {
        "loss": float(loss.detach().item()),
        "recon_loss": float(recon_loss.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
    }
    return loss, metrics

