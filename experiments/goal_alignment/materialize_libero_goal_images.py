from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_LIBERO_ROOT = Path("/data/ztk/code/FastWAM/data/libero_mujoco3.3.2")
DEFAULT_SUITES = [
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_10_no_noops_lerobot",
]
CAMERA_KEYS = ["observation.images.image", "observation.images.wrist_image"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize full LIBERO episode final frames in the goal-alignment metadata format."
    )
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/goal_alignment/libero_episode_last_frames"))
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--max-episodes-per-suite", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_chunk(episode_index: int, chunks_size: int = 1000) -> int:
    return int(episode_index) // int(chunks_size)


def get_video_path(dataset_dir: Path, video_path_template: str, episode_index: int, video_key: str, chunks_size: int) -> Path:
    return dataset_dir / video_path_template.format(
        episode_chunk=episode_chunk(episode_index, chunks_size),
        video_key=video_key,
        episode_index=episode_index,
    )


def read_last_frame(video_path: Path) -> np.ndarray:
    last = None
    errors = []
    try:
        import av

        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                last = np.asarray(frame.to_image().convert("RGB"), dtype=np.uint8)
    except Exception as err:
        errors.append(f"pyav: {err}")

    if last is None:
        try:
            from torchvision.io import read_video

            frames, _, _ = read_video(str(video_path), pts_unit="sec", output_format="THWC")
            if frames.numel() > 0:
                last = frames[-1].cpu().numpy()
        except Exception as err:
            errors.append(f"torchvision: {err}")

    if last is None:
        try:
            for frame in iio.imiter(video_path):
                last = frame
        except Exception as err:
            errors.append(f"imageio: {err}")

    if last is None:
        joined = "\n  ".join(errors)
        raise RuntimeError(
            f"No frames decoded from {video_path}. Tried pyav, torchvision, and imageio.\n"
            f"Install one video backend in the FastWAM environment, e.g. `pip install av` "
            f"or `pip install imageio[ffmpeg]`.\n  {joined}"
        )
    if last.ndim != 3 or last.shape[-1] < 3:
        raise RuntimeError(f"Expected RGB-like frame from {video_path}, got shape {last.shape}")
    return np.asarray(last[..., :3], dtype=np.uint8)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.jsonl"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metadata_path} exists. Pass --overwrite to regenerate.")

    task_to_global_index: dict[str, int] = {}
    rows = []
    for suite in args.suites:
        dataset_dir = args.libero_root / suite
        info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
        episodes = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
        video_template = info["video_path"]
        chunks_size = int(info.get("chunks_size", 1000))

        if args.max_episodes_per_suite is not None:
            episodes = episodes[: int(args.max_episodes_per_suite)]

        for episode in tqdm(episodes, desc=f"Materializing {suite}"):
            episode_index = int(episode["episode_index"])
            task = str(episode["tasks"][0])
            if task not in task_to_global_index:
                task_to_global_index[task] = len(task_to_global_index)
            global_task_index = task_to_global_index[task]

            out_name = f"{suite}_task_{global_task_index:03d}_episode_{episode_index:06d}_final.png"
            out_path = args.output_dir / out_name
            if args.overwrite or not out_path.exists():
                frames = []
                for camera_key in CAMERA_KEYS:
                    video_path = get_video_path(
                        dataset_dir=dataset_dir,
                        video_path_template=video_template,
                        episode_index=episode_index,
                        video_key=camera_key,
                        chunks_size=chunks_size,
                    )
                    frames.append(read_last_frame(video_path))
                composite = np.concatenate(frames, axis=1)
                Image.fromarray(composite).save(out_path)

            rows.append(
                {
                    "task_index": global_task_index,
                    "task": task,
                    "suite": suite,
                    "suite_episode_index": episode_index,
                    "episode_index": len(rows),
                    "frame_index": int(episode["length"]) - 1,
                    "image_path": out_name,
                }
            )

    tmp_metadata = metadata_path.with_suffix(".jsonl.tmp")
    with tmp_metadata.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_metadata.replace(metadata_path)

    tasks_path = args.output_dir / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as f:
        for task, idx in sorted(task_to_global_index.items(), key=lambda item: item[1]):
            f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")

    print(
        f"wrote records={len(rows)} tasks={len(task_to_global_index)} "
        f"images+metadata to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
