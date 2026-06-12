from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset


def _lerobot_action_key(action_meta: dict[str, Any]) -> str:
    key = action_meta["key"]
    return "action" if key == "default" else f"action.{key}"


class ActionOnlyLeRobotDataset(Dataset):
    """Action-only dataset that avoids all video/image decoding."""

    def __init__(
        self,
        dataset_dirs: list[str],
        action_meta: list[dict[str, Any]],
        chunk_len: int = 32,
        global_sample_stride: int = 1,
        val_set_proportion: float = 0.0,
        is_training_set: bool = True,
        seed: int = 42,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ):
        if not dataset_dirs:
            raise ValueError("`dataset_dirs` must not be empty.")
        if not action_meta:
            raise ValueError("`action_meta` must not be empty.")
        self.dataset_dirs = [str(p) for p in dataset_dirs]
        self.action_meta = action_meta
        self.chunk_len = int(chunk_len)
        self.global_sample_stride = int(global_sample_stride)
        self.is_training_set = bool(is_training_set)
        self.mean = mean
        self.std = std

        metas = []
        for ds_dir in self.dataset_dirs:
            ds_root = Path(ds_dir)
            metas.append(LeRobotDatasetMetadata(repo_id=ds_dir, root=ds_root))

        fps_list = [m.fps for m in metas]
        if len(set(fps_list)) != 1:
            raise ValueError(f"All dataset dirs must have the same fps, got {fps_list}.")
        fps = fps_list[0]

        delta_timestamps = {}
        self.action_keys = []
        for meta in action_meta:
            action_key = _lerobot_action_key(meta)
            self.action_keys.append(action_key)
            delta_timestamps[action_key] = [
                (t * self.global_sample_stride) / fps for t in range(self.chunk_len)
            ]

        episodes = {}
        if val_set_proportion < 1e-6:
            for meta in metas:
                episodes[meta.repo_id] = list(range(meta.total_episodes))
        else:
            rng = np.random.default_rng(seed)
            for meta in metas:
                episode_indices = list(range(meta.total_episodes))
                rng.shuffle(episode_indices)
                split_idx = int(meta.total_episodes * (1.0 - val_set_proportion))
                episodes[meta.repo_id] = (
                    episode_indices[:split_idx] if self.is_training_set else episode_indices[split_idx:]
                )

        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            download_videos=False,
        )
        self.multi_dataset.set_during_training(False)

    @property
    def action_dim(self) -> int:
        return int(sum(int(meta["shape"]) for meta in self.action_meta))

    def __len__(self) -> int:
        return len(self.multi_dataset)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.detach().clone().float()
        self.std = std.detach().clone().float().clamp_min(1e-6)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.multi_dataset[idx]
        chunks = []
        pad = None
        for action_key in self.action_keys:
            action = item[action_key]
            if action.ndim == 1:
                action = action.unsqueeze(-1)
            chunks.append(action.float())
            cur_pad = item.get(f"{action_key}_is_pad")
            if cur_pad is not None and pad is None:
                pad = cur_pad.bool()
        action = torch.cat(chunks, dim=-1)
        if action.shape != (self.chunk_len, self.action_dim):
            raise ValueError(
                f"Expected action chunk {(self.chunk_len, self.action_dim)}, got {tuple(action.shape)}."
            )

        if pad is None:
            pad = torch.zeros(self.chunk_len, dtype=torch.bool)

        if self.mean is not None and self.std is not None:
            action = (action - self.mean) / self.std

        return {
            "action": action,
            "action_is_pad": pad,
            "idx": torch.tensor(idx, dtype=torch.long),
        }


@torch.no_grad()
def compute_action_normalization(
    dataset: ActionOnlyLeRobotDataset,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total = torch.zeros(dataset.action_dim, dtype=torch.float64)
    total_sq = torch.zeros(dataset.action_dim, dtype=torch.float64)
    count = torch.zeros((), dtype=torch.float64)

    for batch in loader:
        action = batch["action"].double()
        valid = (~batch["action_is_pad"].bool()).double()
        total += (action * valid.unsqueeze(-1)).sum(dim=(0, 1))
        total_sq += (action.pow(2) * valid.unsqueeze(-1)).sum(dim=(0, 1))
        count += valid.sum()

    count = count.clamp_min(1.0)
    mean = total / count
    var = (total_sq / count) - mean.pow(2)
    std = var.clamp_min(1e-12).sqrt()
    return mean.float(), std.float()

