from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving


DEFAULT_DATASET_DIR = Path("/data/ztk/datasets/libero_goal_image/episode_last_frames")
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
WAN22_TEXT_CACHE_ID = "wan22ti2v5b"


@dataclass(frozen=True)
class GoalImageRecord:
    task_index: int
    task: str
    episode_index: int
    frame_index: int
    image_path: Path
    prompt: str

    @property
    def image_stem(self) -> str:
        return self.image_path.stem


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_cache_path(cache_dir: Path, prompt: str, context_len: int) -> Path:
    return cache_dir / f"{sha256_text(prompt)}.t5_len{context_len}.{WAN22_TEXT_CACHE_ID}.pt"


def image_cache_path(cache_dir: Path, record: GoalImageRecord) -> Path:
    return cache_dir / f"{record.image_stem}.vae.pt"


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def read_metadata(dataset_dir: Path, prompt_template: str = DEFAULT_PROMPT) -> list[GoalImageRecord]:
    metadata_jsonl = dataset_dir / "metadata.jsonl"
    metadata_csv = dataset_dir / "metadata.csv"
    rows: Iterable[dict]
    if metadata_jsonl.exists():
        with metadata_jsonl.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif metadata_csv.exists():
        with metadata_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise FileNotFoundError(f"Expected metadata.jsonl or metadata.csv under {dataset_dir}")

    records = []
    for row in rows:
        rel_image_path = Path(str(row["image_path"]))
        if rel_image_path.parts and rel_image_path.parts[0] == dataset_dir.name:
            image_path = dataset_dir.parent / rel_image_path
        else:
            image_path = dataset_dir / rel_image_path.name
        task = str(row["task"])
        records.append(
            GoalImageRecord(
                task_index=int(row["task_index"]),
                task=task,
                episode_index=int(row["episode_index"]),
                frame_index=int(row["frame_index"]),
                image_path=image_path,
                prompt=prompt_template.format(task=task),
            )
        )
    return records


def preprocess_goal_image(image_path: Path, height: int, width: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    resize = ResizeSmallestSideAspectPreserving(args={"img_w": width, "img_h": height})
    crop = CenterCrop(args={"img_w": width, "img_h": height})
    normalize = Normalize(args={"mean": 0.5, "std": 0.5})
    image_tensor = normalize(crop(resize(image)))  # [C,H,W], range [-1,1]
    return image_tensor.unsqueeze(1).contiguous()  # [C,1,H,W]


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    weights = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class GoalProjector(nn.Module):
    def __init__(
        self,
        text_dim: int = 4096,
        image_dim: int = 48,
        goal_dim: int = 1024,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        num_tasks: int | None = None,
        num_goal_tokens: int = 1,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_goal_tokens = int(num_goal_tokens)
        self.goal_dim = int(goal_dim)
        if self.num_goal_tokens < 1:
            raise ValueError(f"`num_goal_tokens` must be >= 1, got {num_goal_tokens}")
        if self.goal_dim % int(num_heads) != 0:
            raise ValueError(f"`goal_dim` must be divisible by `num_heads`, got {goal_dim} and {num_heads}")

        self.text_queries = nn.Parameter(torch.randn(self.num_goal_tokens, self.goal_dim) * 0.02)
        self.image_queries = nn.Parameter(torch.randn(self.num_goal_tokens, self.goal_dim) * 0.02)
        self.text_input = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, self.goal_dim),
        )
        self.image_input = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, self.goal_dim),
        )
        self.text_attn = nn.MultiheadAttention(
            embed_dim=self.goal_dim,
            num_heads=int(num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.image_attn = nn.MultiheadAttention(
            embed_dim=self.goal_dim,
            num_heads=int(num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.text_ffn = self._make_ffn(self.goal_dim, hidden_dim, dropout)
        self.image_ffn = self._make_ffn(self.goal_dim, hidden_dim, dropout)
        self.text_output_norm = nn.LayerNorm(self.goal_dim)
        self.image_output_norm = nn.LayerNorm(self.goal_dim)
        if num_tasks is None:
            self.text_task_head = None
            self.image_task_head = None
        else:
            self.text_task_head = nn.Linear(self.goal_dim, int(num_tasks))
            self.image_task_head = nn.Linear(self.goal_dim, int(num_tasks))

    @staticmethod
    def _make_ffn(goal_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(goal_dim),
            nn.Linear(goal_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, goal_dim),
        )

    def encode_text(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = int(context.shape[0])
        queries = self.text_queries.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = self.text_input(context)
        key_padding_mask = None if mask is None else ~mask.to(device=context.device, dtype=torch.bool)
        attended, _ = self.text_attn(
            query=queries,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.text_output_norm(attended + self.text_ffn(attended))

    def encode_image(self, vae_latent: torch.Tensor) -> torch.Tensor:
        if vae_latent.ndim == 5:
            batch_size = int(vae_latent.shape[0])
            tokens = vae_latent.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, vae_latent.shape[1])
        elif vae_latent.ndim == 4:
            tokens = vae_latent.permute(1, 2, 3, 0).reshape(1, -1, vae_latent.shape[0])
            batch_size = 1
        else:
            raise ValueError(f"Expected VAE latent [B,C,T,H,W] or [C,T,H,W], got {tuple(vae_latent.shape)}")
        queries = self.image_queries.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = self.image_input(tokens)
        attended, _ = self.image_attn(
            query=queries,
            key=tokens,
            value=tokens,
            need_weights=False,
        )
        return self.image_output_norm(attended + self.image_ffn(attended))


def pool_goal_tokens(goal: torch.Tensor) -> torch.Tensor:
    if goal.ndim == 2:
        return goal
    if goal.ndim != 3:
        raise ValueError(f"Expected goal tokens [B,D] or [B,M,D], got {tuple(goal.shape)}")
    return goal.mean(dim=1)


def alignment_loss(
    text_goal: torch.Tensor,
    image_goal: torch.Tensor,
    task_labels: torch.Tensor | None = None,
    text_task_logits: torch.Tensor | None = None,
    image_task_logits: torch.Tensor | None = None,
    cosine_weight: float = 1.0,
    mse_weight: float = 0.1,
    variance_weight: float = 0.1,
    task_weight: float = 0.5,
    variance_target_std: float = 1.0,
    set_weight: float = 1.0,
    diversity_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    text_tokens = text_goal.unsqueeze(1) if text_goal.ndim == 2 else text_goal
    image_tokens = image_goal.unsqueeze(1) if image_goal.ndim == 2 else image_goal
    text_pooled = pool_goal_tokens(text_tokens)
    image_pooled = pool_goal_tokens(image_tokens)
    text_norm = F.normalize(text_pooled.float(), dim=-1)
    image_norm = F.normalize(image_pooled.float(), dim=-1)
    loss_cos = 1.0 - (text_norm * image_norm).sum(dim=-1).mean()
    loss_mse = F.mse_loss(text_pooled.float(), image_pooled.float())

    token_sim = torch.einsum(
        "bmd,bnd->bmn",
        F.normalize(text_tokens.float(), dim=-1),
        F.normalize(image_tokens.float(), dim=-1),
    )
    loss_set = 1.0 - 0.5 * (token_sim.max(dim=2).values.mean() + token_sim.max(dim=1).values.mean())

    def _diversity_loss(z: torch.Tensor) -> torch.Tensor:
        if z.shape[1] <= 1:
            return z.new_tensor(0.0)
        z = F.normalize(z.float(), dim=-1)
        sim = torch.einsum("bmd,bnd->bmn", z, z)
        mask = ~torch.eye(z.shape[1], dtype=torch.bool, device=z.device).unsqueeze(0)
        return F.relu(sim[mask.expand_as(sim)]).mean()

    def _variance_loss(z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 3:
            z = z.reshape(-1, z.shape[-1])
        if z.shape[0] <= 1:
            return z.new_tensor(0.0)
        std = torch.sqrt(z.float().var(dim=0, unbiased=False) + 1e-4)
        return F.relu(float(variance_target_std) - std).mean()

    loss_var_text = _variance_loss(text_tokens)
    loss_var_image = _variance_loss(image_tokens)
    loss_div_text = _diversity_loss(text_tokens)
    loss_div_image = _diversity_loss(image_tokens)
    loss_task_text = text_tokens.new_tensor(0.0)
    loss_task_image = image_tokens.new_tensor(0.0)
    text_task_acc = float("nan")
    image_task_acc = float("nan")
    if task_labels is not None:
        labels = task_labels.to(device=text_tokens.device, dtype=torch.long)
        if text_task_logits is not None:
            loss_task_text = F.cross_entropy(text_task_logits.float(), labels)
            text_task_acc = float((text_task_logits.argmax(dim=-1) == labels).float().mean().detach().item())
        if image_task_logits is not None:
            loss_task_image = F.cross_entropy(image_task_logits.float(), labels)
            image_task_acc = float((image_task_logits.argmax(dim=-1) == labels).float().mean().detach().item())

    loss = (
        cosine_weight * loss_cos
        + mse_weight * loss_mse
        + set_weight * loss_set
        + variance_weight * (loss_var_text + loss_var_image)
        + diversity_weight * (loss_div_text + loss_div_image)
        + task_weight * (loss_task_text + loss_task_image)
    )
    metrics = {
        "loss": float(loss.detach().item()),
        "loss_cos": float(loss_cos.detach().item()),
        "loss_mse": float(loss_mse.detach().item()),
        "loss_set": float(loss_set.detach().item()),
        "loss_var_text": float(loss_var_text.detach().item()),
        "loss_var_image": float(loss_var_image.detach().item()),
        "loss_div_text": float(loss_div_text.detach().item()),
        "loss_div_image": float(loss_div_image.detach().item()),
        "loss_task_text": float(loss_task_text.detach().item()),
        "loss_task_image": float(loss_task_image.detach().item()),
        "cosine": float((text_norm * image_norm).sum(dim=-1).mean().detach().item()),
        "set_cosine": float((1.0 - loss_set).detach().item()),
        "text_token_diversity": float(loss_div_text.detach().item()),
        "image_token_diversity": float(loss_div_image.detach().item()),
        "text_task_acc": text_task_acc,
        "image_task_acc": image_task_acc,
    }
    return loss, metrics


class CachedGoalAlignmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        text_cache_dir: Path,
        image_cache_dir: Path,
        context_len: int,
        prompt_template: str = DEFAULT_PROMPT,
    ):
        self.records = read_metadata(dataset_dir, prompt_template=prompt_template)
        self.text_cache_dir = text_cache_dir
        self.image_cache_dir = image_cache_dir
        self.context_len = int(context_len)

        missing = []
        for record in self.records:
            if not text_cache_path(text_cache_dir, record.prompt, self.context_len).exists():
                missing.append(str(text_cache_path(text_cache_dir, record.prompt, self.context_len)))
            if not image_cache_path(image_cache_dir, record).exists():
                missing.append(str(image_cache_path(image_cache_dir, record)))
            if len(missing) >= 8:
                break
        if missing:
            joined = "\n".join(missing)
            raise FileNotFoundError(f"Missing latent cache files, first examples:\n{joined}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        text_payload = torch.load(
            text_cache_path(self.text_cache_dir, record.prompt, self.context_len),
            map_location="cpu",
            weights_only=False,
        )
        image_payload = torch.load(
            image_cache_path(self.image_cache_dir, record),
            map_location="cpu",
            weights_only=False,
        )
        return {
            "context": text_payload["context"].float(),
            "mask": text_payload["mask"].bool(),
            "vae_latent": image_payload["latent"].float(),
            "task_index": record.task_index,
            "episode_index": record.episode_index,
        }
