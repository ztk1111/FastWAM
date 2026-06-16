from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_PROMPT,
    CachedGoalAlignmentDataset,
    GoalProjector,
    pool_goal_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick collapse checks for goal alignment checkpoints.")
    parser.add_argument("--ckpt", type=Path, default=Path("runs/goal_alignment/alignment/best.pt"))
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--text-cache-dir", type=Path, default=None)
    parser.add_argument("--image-cache-dir", type=Path, default=None)
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt-template", type=str, default=None)
    return parser.parse_args()


def offdiag_cosine(z: torch.Tensor) -> float:
    z = F.normalize(z.float(), dim=-1)
    sim = z @ z.T
    n = sim.shape[0]
    if n <= 1:
        return float("nan")
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    return float(sim[mask].mean().item())


def nearest_task_accuracy(query: torch.Tensor, keys: torch.Tensor, labels: torch.Tensor) -> float:
    query = F.normalize(query.float(), dim=-1)
    keys = F.normalize(keys.float(), dim=-1)
    sim = query @ keys.T
    sim.fill_diagonal_(-float("inf"))
    pred = labels[sim.argmax(dim=1)]
    return float((pred == labels).float().mean().item())


def centroid_accuracy(query: torch.Tensor, reference: torch.Tensor, labels: torch.Tensor) -> float:
    query = F.normalize(query.float(), dim=-1)
    reference = F.normalize(reference.float(), dim=-1)
    task_ids = labels.unique(sorted=True)
    centroids = []
    for task_id in task_ids:
        centroids.append(reference[labels == task_id].mean(dim=0))
    centroids = F.normalize(torch.stack(centroids, dim=0), dim=-1)
    pred = task_ids[(query @ centroids.T).argmax(dim=1)]
    return float((pred == labels).float().mean().item())


def mean_token_offdiag_cosine(tokens: torch.Tensor) -> float:
    if tokens.ndim != 3:
        return float("nan")
    vals = []
    for token_idx in range(tokens.shape[1]):
        vals.append(offdiag_cosine(tokens[:, token_idx, :]))
    return float(torch.tensor(vals).mean().item())


def token_index_alignment(text_tokens: torch.Tensor, image_tokens: torch.Tensor) -> float:
    if text_tokens.ndim != 3 or image_tokens.ndim != 3:
        return float("nan")
    sim = (F.normalize(text_tokens.float(), dim=-1) * F.normalize(image_tokens.float(), dim=-1)).sum(dim=-1)
    return float(sim.mean().item())


def within_sample_token_cosine(tokens: torch.Tensor) -> float:
    if tokens.ndim != 3 or tokens.shape[1] <= 1:
        return float("nan")
    z = F.normalize(tokens.float(), dim=-1)
    sim = torch.einsum("bmd,bnd->bmn", z, z)
    mask = ~torch.eye(tokens.shape[1], dtype=torch.bool, device=sim.device).unsqueeze(0)
    return float(sim[mask.expand_as(sim)].mean().item())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    dataset_dir = Path(args.dataset_dir or cfg.get("dataset_dir", DEFAULT_DATASET_DIR))
    text_cache_dir = Path(args.text_cache_dir or cfg.get("text_cache_dir", "runs/goal_alignment/text_latents"))
    image_cache_dir = Path(args.image_cache_dir or cfg.get("image_cache_dir", "runs/goal_alignment/image_latents"))
    context_len = int(args.context_len or cfg.get("context_len", 128))
    prompt_template = args.prompt_template or cfg.get("prompt_template", DEFAULT_PROMPT)

    model = GoalProjector(
        text_dim=4096,
        image_dim=48,
        goal_dim=int(cfg.get("goal_dim", 1024)),
        hidden_dim=int(cfg.get("hidden_dim", 2048)),
        dropout=0.0,
        num_tasks=cfg.get("num_tasks"),
        num_goal_tokens=int(cfg.get("num_goal_tokens", 1)),
        num_heads=int(cfg.get("num_heads", 8)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    dataset = CachedGoalAlignmentDataset(
        dataset_dir=dataset_dir,
        text_cache_dir=text_cache_dir,
        image_cache_dir=image_cache_dir,
        context_len=context_len,
        prompt_template=prompt_template,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    text_goals = []
    image_goals = []
    task_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding goals"):
            context = batch["context"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device=device, dtype=torch.bool)
            latent = batch["vae_latent"].to(device=device, dtype=torch.float32)
            text_goals.append(model.encode_text(context, mask).cpu())
            image_goals.append(model.encode_image(latent).cpu())
            task_labels.append(batch["task_index"].long())

    text_goal_tokens = torch.cat(text_goals, dim=0)
    image_goal_tokens = torch.cat(image_goals, dim=0)
    text_goals = pool_goal_tokens(text_goal_tokens)
    image_goals = pool_goal_tokens(image_goal_tokens)
    labels = torch.cat(task_labels, dim=0)
    aligned_cos = float((F.normalize(text_goals, dim=-1) * F.normalize(image_goals, dim=-1)).sum(dim=-1).mean())

    token_sim = torch.einsum(
        "bmd,bnd->bmn",
        F.normalize(text_goal_tokens.float(), dim=-1),
        F.normalize(image_goal_tokens.float(), dim=-1),
    ) if text_goal_tokens.ndim == 3 and image_goal_tokens.ndim == 3 else None
    set_cos = float("nan") if token_sim is None else float(
        0.5 * (token_sim.max(dim=2).values.mean() + token_sim.max(dim=1).values.mean()).item()
    )

    print(f"samples: {text_goals.shape[0]}")
    print(f"tasks: {labels.unique().numel()}")
    print(f"dataset_dir: {dataset_dir}")
    print(f"text_cache_dir: {text_cache_dir}")
    print(f"image_cache_dir: {image_cache_dir}")
    print(f"goal token shape: {tuple(text_goal_tokens.shape[1:])}")
    print(f"aligned pooled text-image cosine: {aligned_cos:.4f}")
    print(f"set text-image cosine: {set_cos:.4f}")
    print(f"text offdiag cosine: {offdiag_cosine(text_goals):.4f}")
    print(f"image offdiag cosine: {offdiag_cosine(image_goals):.4f}")
    print(f"text pooled mean feature std: {text_goals.std(dim=0).mean().item():.6f}")
    print(f"image pooled mean feature std: {image_goals.std(dim=0).mean().item():.6f}")
    if text_goal_tokens.ndim == 3:
        print(f"text token mean feature std: {text_goal_tokens.reshape(-1, text_goal_tokens.shape[-1]).std(dim=0).mean().item():.6f}")
        print(f"image token mean feature std: {image_goal_tokens.reshape(-1, image_goal_tokens.shape[-1]).std(dim=0).mean().item():.6f}")
        print(f"text token-index offdiag cosine: {mean_token_offdiag_cosine(text_goal_tokens):.4f}")
        print(f"image token-index offdiag cosine: {mean_token_offdiag_cosine(image_goal_tokens):.4f}")
        print(f"token-index text-image cosine: {token_index_alignment(text_goal_tokens, image_goal_tokens):.4f}")
        print(f"text within-sample token cosine: {within_sample_token_cosine(text_goal_tokens):.4f}")
        print(f"image within-sample token cosine: {within_sample_token_cosine(image_goal_tokens):.4f}")
    print(f"text nearest same-task acc: {nearest_task_accuracy(text_goals, text_goals, labels):.4f}")
    print(f"image nearest same-task acc: {nearest_task_accuracy(image_goals, image_goals, labels):.4f}")
    print(f"cross text->image same-task acc: {nearest_task_accuracy(text_goals, image_goals, labels):.4f}")
    print(f"text->image centroid task acc: {centroid_accuracy(text_goals, image_goals, labels):.4f}")
    print(f"image->text centroid task acc: {centroid_accuracy(image_goals, text_goals, labels):.4f}")


if __name__ == "__main__":
    main()
