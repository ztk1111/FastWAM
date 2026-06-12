from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class CachedRobotVideoDataset(Dataset):
    """Read preprocessed FastWAM training samples from tensor shards.

    The returned sample schema matches ``RobotVideoDataset`` so the existing
    trainer/model path can be reused without video decoding at training time.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        video_dtype: str = "float32",
        context_dtype: str | None = None,
        action_dtype: str | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        index_path = self.cache_dir / "index.pt"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing cache metadata: {meta_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Missing cache index: {index_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta: dict[str, Any] = json.load(f)
        self.index = torch.load(index_path, map_location="cpu", weights_only=False)
        self.shard_ids = self.index["shard_ids"].long()
        self.offsets = self.index["offsets"].long()
        self.source_indices = self.index.get("source_indices")

        self.video_dtype = _dtype_from_name(video_dtype)
        self.context_dtype = _dtype_from_name(context_dtype) if context_dtype is not None else None
        self.action_dtype = _dtype_from_name(action_dtype) if action_dtype is not None else None

        self._loaded_shard_id: int | None = None
        self._loaded_shard: dict[str, Any] | None = None

    def __len__(self) -> int:
        return int(self.shard_ids.numel())

    def _load_shard(self, shard_id: int) -> dict[str, Any]:
        if self._loaded_shard_id == shard_id and self._loaded_shard is not None:
            return self._loaded_shard
        shard_path = self.cache_dir / "shards" / f"shard_{shard_id:06d}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing cache shard: {shard_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        self._loaded_shard_id = shard_id
        self._loaded_shard = shard
        return shard

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_id = int(self.shard_ids[idx].item())
        offset = int(self.offsets[idx].item())
        shard = self._load_shard(shard_id)

        video = shard["video"][offset]
        if video.dtype == torch.uint8:
            video = video.to(dtype=self.video_dtype).div(127.5).sub(1.0)
        else:
            video = video.to(dtype=self.video_dtype)

        sample: dict[str, Any] = {
            "video": video,
            "action": shard["action"][offset],
            "proprio": shard["proprio"][offset],
            "prompt": shard["prompt"][offset],
            "context": shard["context"][offset],
            "context_mask": shard["context_mask"][offset].bool(),
            "image_is_pad": shard["image_is_pad"][offset].bool(),
            "action_is_pad": shard["action_is_pad"][offset].bool(),
            "proprio_is_pad": shard["proprio_is_pad"][offset].bool(),
        }
        if self.action_dtype is not None:
            sample["action"] = sample["action"].to(dtype=self.action_dtype)
            sample["proprio"] = sample["proprio"].to(dtype=self.action_dtype)
        if self.context_dtype is not None:
            sample["context"] = sample["context"].to(dtype=self.context_dtype)
        if "idx" in shard:
            sample["idx"] = shard["idx"][offset]
        elif self.source_indices is not None:
            sample["idx"] = self.source_indices[idx]
        else:
            sample["idx"] = torch.tensor(idx, dtype=torch.long)
        return sample


def _dtype_from_name(name: str | None) -> torch.dtype:
    if name is None:
        return torch.float32
    normalized = str(name).lower()
    mapping = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[normalized]

