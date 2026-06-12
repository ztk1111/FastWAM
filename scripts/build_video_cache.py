#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.datasets.cache.build_robot_video_cache import build_robot_video_cache
from fastwam.runtime import build_datasets
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    cache_cfg = cfg.get("cache")
    if cache_cfg is None or cache_cfg.get("output_dir") is None:
        raise ValueError(
            "Set +cache.output_dir, for example: "
            "+cache.output_dir=./data/cache/libero_idm_2cam224"
        )

    output_root = Path(str(cache_cfg.output_dir))
    split = str(cache_cfg.get("split", "train"))
    shard_size = int(cache_cfg.get("shard_size", 128))
    max_samples = cache_cfg.get("max_samples")
    max_samples = None if max_samples is None else int(max_samples)
    start_index = int(cache_cfg.get("start_index", 0))
    overwrite = bool(cache_cfg.get("overwrite", False))
    store_video = str(cache_cfg.get("store_video", "uint8"))
    tensor_dtype = str(cache_cfg.get("tensor_dtype", "float16"))

    misc.register_work_dir(str(output_root / "_work"))

    if split == "train":
        dataset = instantiate(cfg.data.train)
    elif split == "val":
        _, dataset = build_datasets(cfg.data)
    else:
        raise ValueError(f"Unsupported cache split: {split}. Expected 'train' or 'val'.")

    out_dir = output_root / split
    meta = build_robot_video_cache(
        dataset=dataset,
        output_dir=out_dir,
        shard_size=shard_size,
        max_samples=max_samples,
        start_index=start_index,
        overwrite=overwrite,
        store_video=store_video,
        tensor_dtype=tensor_dtype,
    )
    print(f"[cache] wrote {meta['num_samples']} samples to {out_dir}")
    if meta["num_failed"]:
        print(f"[cache] skipped {meta['num_failed']} failed samples; see {out_dir / 'failed.json'}")


if __name__ == "__main__":
    main()

