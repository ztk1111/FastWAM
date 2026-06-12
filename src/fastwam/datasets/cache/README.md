# FastWAM video cache

This cache materializes `RobotVideoDataset` samples into tensor shards so
training does not open videos or call PyAV/torchvision decoders in dataloader
workers.

## Build

```bash
PYTHONPATH=src python scripts/build_video_cache.py \
  task=libero_idm_2cam224_1e-4 \
  +cache.output_dir=./data/cache/libero_idm_2cam224 \
  +cache.split=train \
  +cache.shard_size=128 \
  +cache.store_video=uint8 \
  +cache.tensor_dtype=float16
```

The builder still decodes videos once during preprocessing. The resulting
training dataset reads only `.pt` shards.

Useful options:

- `+cache.max_samples=1000`: build a small cache for testing.
- `+cache.start_index=10000`: start from a later source sample.
- `+cache.overwrite=true`: rebuild an existing cache directory.
- `+cache.store_video=uint8`: lowest disk usage; loaded as `[-1, 1]` float.
- `+cache.tensor_dtype=float16`: store action/proprio/text context in fp16.

## Train

After building `./data/cache/libero_idm_2cam224/train`, switch the data config:

```bash
PYTHONPATH=src python scripts/train.py \
  task=libero_idm_2cam224_1e-4 \
  data=libero_2cam_cached \
  num_workers=4
```

For the existing shell launcher, pass `data=libero_2cam_cached` as an override.

Start with `num_workers=2` or `4`. Each worker keeps one shard loaded, so larger
`shard_size` values trade fewer files for more CPU memory.

## Disk estimate

For LIBERO 2-camera 224x224 with 9 sampled video frames, `uint8` video storage is
about 2.7 MB per sample before shard overhead. Full LIBERO-scale caches can be
hundreds of GB. Build a small cache first and check the real size with `du -sh`.
