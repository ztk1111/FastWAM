from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def build_robot_video_cache(
    dataset: Dataset,
    output_dir: str | Path,
    shard_size: int = 128,
    max_samples: int | None = None,
    start_index: int = 0,
    overwrite: bool = False,
    store_video: str = "uint8",
    tensor_dtype: str = "float16",
) -> dict[str, Any]:
    """Materialize a RobotVideoDataset-compatible dataset into tensor shards."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Cache directory already exists: {output_dir}. Set cache.overwrite=true.")
        shutil.rmtree(output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    total_len = len(dataset)
    end_index = total_len if max_samples is None else min(total_len, start_index + int(max_samples))
    if start_index < 0 or start_index >= total_len:
        raise ValueError(f"`start_index` out of bounds: {start_index} for dataset length {total_len}")
    if end_index <= start_index:
        raise ValueError(f"No samples to cache: start={start_index}, end={end_index}")

    float_dtype = _dtype_from_name(tensor_dtype)
    shard_samples: list[dict[str, Any]] = []
    shard_ids: list[int] = []
    offsets: list[int] = []
    source_indices: list[int] = []
    failed: list[dict[str, Any]] = []
    shard_id = 0

    for source_idx in tqdm(range(start_index, end_index), desc=f"building cache -> {output_dir}"):
        try:
            sample = dataset[source_idx]
        except Exception as err:
            failed.append({"idx": int(source_idx), "error": repr(err)})
            continue
        shard_ids.append(shard_id)
        offsets.append(len(shard_samples))
        source_indices.append(int(source_idx))
        shard_samples.append(_pack_sample(sample, store_video=store_video, float_dtype=float_dtype))

        if len(shard_samples) >= shard_size:
            _save_shard(shards_dir, shard_id, shard_samples)
            shard_id += 1
            shard_samples = []

    if shard_samples:
        _save_shard(shards_dir, shard_id, shard_samples)
        shard_id += 1

    index = {
        "shard_ids": torch.tensor(shard_ids, dtype=torch.long),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "source_indices": torch.tensor(source_indices, dtype=torch.long),
    }
    _atomic_torch_save(index, output_dir / "index.pt")

    meta = {
        "format_version": 1,
        "source_dataset": type(dataset).__name__,
        "source_length": int(total_len),
        "start_index": int(start_index),
        "end_index": int(end_index),
        "num_samples": int(len(source_indices)),
        "num_failed": int(len(failed)),
        "num_shards": int(shard_id),
        "shard_size": int(shard_size),
        "store_video": str(store_video),
        "tensor_dtype": str(tensor_dtype),
    }
    _atomic_json_save(meta, output_dir / "meta.json")
    if failed:
        _atomic_json_save(failed, output_dir / "failed.json")
    return meta


def _pack_sample(sample: dict[str, Any], store_video: str, float_dtype: torch.dtype) -> dict[str, Any]:
    video = sample["video"].detach().cpu()
    if store_video == "uint8":
        video = ((video.float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    elif store_video == "float16":
        video = video.to(torch.float16)
    elif store_video == "float32":
        video = video.to(torch.float32)
    else:
        raise ValueError(f"Unsupported `store_video`: {store_video}")

    return {
        "video": video.contiguous(),
        "action": sample["action"].detach().cpu().to(float_dtype).contiguous(),
        "proprio": sample["proprio"].detach().cpu().to(float_dtype).contiguous(),
        "prompt": str(sample["prompt"]),
        "context": sample["context"].detach().cpu().to(float_dtype).contiguous(),
        "context_mask": sample["context_mask"].detach().cpu().bool().contiguous(),
        "image_is_pad": sample["image_is_pad"].detach().cpu().bool().contiguous(),
        "action_is_pad": sample["action_is_pad"].detach().cpu().bool().contiguous(),
        "proprio_is_pad": sample["proprio_is_pad"].detach().cpu().bool().contiguous(),
        "idx": torch.as_tensor(sample.get("idx", -1), dtype=torch.long),
    }


def _save_shard(shards_dir: Path, shard_id: int, samples: list[dict[str, Any]]):
    shard: dict[str, Any] = {
        "video": torch.stack([s["video"] for s in samples], dim=0),
        "action": torch.stack([s["action"] for s in samples], dim=0),
        "proprio": torch.stack([s["proprio"] for s in samples], dim=0),
        "prompt": [s["prompt"] for s in samples],
        "context": torch.stack([s["context"] for s in samples], dim=0),
        "context_mask": torch.stack([s["context_mask"] for s in samples], dim=0),
        "image_is_pad": torch.stack([s["image_is_pad"] for s in samples], dim=0),
        "action_is_pad": torch.stack([s["action_is_pad"] for s in samples], dim=0),
        "proprio_is_pad": torch.stack([s["proprio_is_pad"] for s in samples], dim=0),
        "idx": torch.stack([s["idx"] for s in samples], dim=0),
    }
    _atomic_torch_save(shard, shards_dir / f"shard_{shard_id:06d}.pt")


def _atomic_torch_save(obj: Any, path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _atomic_json_save(obj: Any, path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported tensor dtype: {name}")
    return mapping[normalized]

