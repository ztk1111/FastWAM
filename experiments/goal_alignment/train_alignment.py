from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_PROMPT,
    CachedGoalAlignmentDataset,
    GoalProjector,
    alignment_loss,
    pool_goal_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight text/final-image goal latent alignment heads.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--text-cache-dir", type=Path, default=Path("runs/goal_alignment/text_latents"))
    parser.add_argument("--image-cache-dir", type=Path, default=Path("runs/goal_alignment/image_latents"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/goal_alignment/alignment"))
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--goal-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-goal-tokens", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.0)
    parser.add_argument("--variance-weight", type=float, default=0.01)
    parser.add_argument("--variance-target-std", type=float, default=1.0)
    parser.add_argument("--task-weight", type=float, default=0.5)
    parser.add_argument("--set-weight", type=float, default=1.0)
    parser.add_argument("--diversity-weight", type=float, default=0.6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT)
    return parser.parse_args()


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    return indices[n_val:], indices[:n_val]


def run_epoch(
    model: GoalProjector,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    cosine_weight: float,
    mse_weight: float,
    variance_weight: float,
    variance_target_std: float,
    task_weight: float,
    set_weight: float,
    diversity_weight: float,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    sums = {
        "loss": 0.0,
        "loss_cos": 0.0,
        "loss_mse": 0.0,
        "loss_set": 0.0,
        "loss_var_text": 0.0,
        "loss_var_image": 0.0,
        "loss_div_text": 0.0,
        "loss_div_image": 0.0,
        "loss_task_text": 0.0,
        "loss_task_image": 0.0,
        "cosine": 0.0,
        "set_cosine": 0.0,
        "text_token_diversity": 0.0,
        "image_token_diversity": 0.0,
        "text_task_acc": 0.0,
        "image_task_acc": 0.0,
    }
    count = 0
    context_manager = torch.enable_grad() if train else torch.no_grad()
    with context_manager:
        for batch in loader:
            context = batch["context"].to(device=device, dtype=torch.float32, non_blocking=True)
            mask = batch["mask"].to(device=device, dtype=torch.bool, non_blocking=True)
            latent = batch["vae_latent"].to(device=device, dtype=torch.float32, non_blocking=True)
            task_labels = batch["task_index"].to(device=device, dtype=torch.long, non_blocking=True)
            text_goal = model.encode_text(context, mask)
            image_goal = model.encode_image(latent)
            text_pooled = pool_goal_tokens(text_goal)
            image_pooled = pool_goal_tokens(image_goal)
            text_task_logits = model.text_task_head(text_pooled) if model.text_task_head is not None else None
            image_task_logits = model.image_task_head(image_pooled) if model.image_task_head is not None else None
            loss, metrics = alignment_loss(
                text_goal=text_goal,
                image_goal=image_goal,
                task_labels=task_labels,
                text_task_logits=text_task_logits,
                image_task_logits=image_task_logits,
                cosine_weight=cosine_weight,
                mse_weight=mse_weight,
                variance_weight=variance_weight,
                task_weight=task_weight,
                variance_target_std=variance_target_std,
                set_weight=set_weight,
                diversity_weight=diversity_weight,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = int(context.shape[0])
            count += batch_size
            for key in sums:
                sums[key] += metrics[key] * batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CachedGoalAlignmentDataset(
        dataset_dir=args.dataset_dir,
        text_cache_dir=args.text_cache_dir,
        image_cache_dir=args.image_cache_dir,
        context_len=args.context_len,
        prompt_template=args.prompt_template,
    )
    num_tasks = max(record.task_index for record in dataset.records) + 1
    train_idx, val_idx = split_indices(len(dataset), args.val_ratio, args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = GoalProjector(
        text_dim=4096,
        image_dim=48,
        goal_dim=args.goal_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_tasks=num_tasks,
        num_goal_tokens=args.num_goal_tokens,
        num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["dataset_size"] = len(dataset)
    config["train_size"] = len(train_idx)
    config["val_size"] = len(val_idx)
    config["num_tasks"] = num_tasks
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    history_path = args.output_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    best_val = float("inf")
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training alignment"):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
            variance_weight=args.variance_weight,
            variance_target_std=args.variance_target_std,
            task_weight=args.task_weight,
            set_weight=args.set_weight,
            diversity_weight=args.diversity_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
            variance_weight=args.variance_weight,
            variance_target_std=args.variance_target_std,
            task_weight=args.task_weight,
            set_weight=args.set_weight,
            diversity_weight=args.diversity_weight,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        latest_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(latest_payload, args.output_dir / "latest.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(latest_payload, args.output_dir / "best.pt")

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_metrics['loss']:.4f} train_cos={train_metrics['cosine']:.4f} "
                f"train_set={train_metrics['set_cosine']:.4f} "
                f"train_acc=({train_metrics['text_task_acc']:.3f},{train_metrics['image_task_acc']:.3f}) "
                f"val_loss={val_metrics['loss']:.4f} val_cos={val_metrics['cosine']:.4f} "
                f"val_set={val_metrics['set_cosine']:.4f} "
                f"val_acc=({val_metrics['text_task_acc']:.3f},{val_metrics['image_task_acc']:.3f})"
            )

    print(f"done best_val_loss={best_val:.4f} output={args.output_dir}")


if __name__ == "__main__":
    main()
