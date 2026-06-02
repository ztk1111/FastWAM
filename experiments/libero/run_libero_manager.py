"""
LIBERO 多 GPU 多任务评估管理模块 (Manager)。

该模块使用 Hydra 配置驱动，通过 shell 脚本 (run_libero_parallel_test.sh)
在多 GPU 环境下并行调度 LIBERO 任务的评估。

核心功能:
1. 创建任务列表文件：通过 LIBERO benchmark API 枚举指定套件的所有任务。
2. 配置传递：将 Hydra 配置参数通过环境变量传递给并行执行脚本。
3. 多 GPU 调度：由 bash 脚本实现 GPU 任务分配和并行执行。
4. 任务列表管理：支持 create_only 模式仅创建任务列表不执行评估。

与 RoboTwin Manager 不同，LIBERO Manager 的并行调度由外部 bash 脚本实现，
Python 端主要负责配置组织和任务列表生成。
"""

import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf


def create_task_file(output_file: Path, task_suite_names: list[str]) -> Path:
    """创建包含所有待评估任务的任务列表文件。

    通过 LIBERO benchmark API 获取指定套件的所有任务，
    每行格式为 "<套件名>,<任务ID>"，如 "libero_spatial,0"。

    Args:
        output_file: 输出任务列表文件路径。
        task_suite_names: 任务套件名称列表（如 ["libero_spatial", "libero_object"]）。

    Returns:
        Path: 创建的任务列表文件路径。

    输出文件示例 (tasks.txt):
        libero_spatial,0
        libero_spatial,1
        libero_object,0
        ...
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    with output_file.open("w", encoding="utf-8") as f:
        for suite_name in task_suite_names:
            task_suite = benchmark_dict[suite_name]()
            n_tasks = int(task_suite.n_tasks)
            print(f"\n{suite_name}:")
            print(f"- Number of tasks: {n_tasks}")
            for task_id in range(n_tasks):
                f.write(f"{suite_name},{task_id}\n")
                total_tasks += 1

    print(f"\nTask list created: {output_file}")
    print(f"Total tasks: {total_tasks}")
    return output_file


def _is_blocked_override(raw_override: str) -> bool:
    """判断 Hydra override 是否应被屏蔽（不传递给 worker 进程）。

    被屏蔽的 override 类型:
        - 任务级特定参数 (task, ckpt, gpu_id 等)
        - MULTIRUN.* 和 hydra.* 参数（管理器专用）

    Args:
        raw_override: 原始 Hydra override 字符串。

    Returns:
        bool: 如果应屏蔽则返回 True。
    """
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    """收集所有非屏蔽的 Hydra override，传递给 worker 进程。

    Returns:
        list[str]: 需要传递给 worker 的 override 列表。
    """
    hydra_overrides = list(HydraConfig.get().overrides.task)
    return [ov for ov in hydra_overrides if not _is_blocked_override(ov)]


def _resolve_worker_task_choice() -> str:
    """解析 Hydra 运行时选择的 task 配置。

    从 Hydra 运行时中获取 task 选择，例如 "world_action_model_forward_224"。

    Returns:
        str: task 配置名称。

    Raises:
        ValueError: 如果 task 选择为空则抛出。
    """
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError(
            "Hydra task choice is empty. Please pass task=... (e.g., task=world_action_model_forward_224)."
        )
    return str(task_choice)


def run_evaluation(
    *,
    task_file: Path,
    task_choice: str,
    ckpt: str,
    num_gpus: int,
    num_trials: int,
    max_tasks_per_gpu: int,
    output_dir: Path,
    extra_overrides: list[str],
) -> None:
    """启动 LIBERO 多 GPU 并行评估脚本。

    通过设置环境变量将配置传递给 bash 脚本 (run_libero_parallel_test.sh)，
    由脚本负责实际的 GPU 调度和任务分配。

    传递的环境变量:
        CONFIG: task 配置名称。
        CKPT: 检查点路径。
        NUM_GPUS: GPU 数量。
        NUM_TRIALS: 每任务 episode 数。
        MAX_TASKS_PER_GPU: 每 GPU 最大并发任务数。
        ROOT_DIR: 项目根目录。
        RUN_ID: 运行标识（时间戳）。
        OUTPUT_DIR: 输出目录。
        EXTRA_ARGS: 额外的命令行参数。
        EXP_NAME: 实验名称（可选）。

    Args:
        task_file: 任务列表文件路径。
        task_choice: Hydra task 配置名称。
        ckpt: 模型检查点路径。
        num_gpus: 使用的 GPU 数量。
        num_trials: 每个任务运行的 episode 数。
        max_tasks_per_gpu: 每 GPU 最大并发任务数。
        output_dir: 输出目录。
        extra_overrides: 额外的 Hydra override 参数列表。

    Raises:
        FileNotFoundError: 评估脚本不存在时抛出。
        subprocess.CalledProcessError: 脚本执行失败时抛出。
    """
    script_path = Path("experiments/libero/run_libero_parallel_test.sh")
    if not script_path.exists():
        raise FileNotFoundError(f"Evaluation script not found: {script_path}")

    root_dir = os.getcwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = shlex.join(extra_overrides) if extra_overrides else ""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = os.environ.copy()
    env.update(
        {
            "CONFIG": task_choice,
            "CKPT": ckpt,
            "NUM_GPUS": str(num_gpus),
            "NUM_TRIALS": str(num_trials),
            "MAX_TASKS_PER_GPU": str(max_tasks_per_gpu),
            "ROOT_DIR": root_dir,
            "RUN_ID": run_id,
            "OUTPUT_DIR": str(output_dir),
            "EXTRA_ARGS": extra_args,
            "EXP_NAME": os.environ.get("EXP_NAME", ""),
        }
    )

    print("\nStarting evaluation (Hydra manager)...")
    print(f"task: {task_choice}")
    print(f"Checkpoint: {ckpt}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Trials per task: {num_trials}")
    print(f"Max tasks per GPU: {max_tasks_per_gpu}")
    print(f"Output directory: {output_dir}")
    if extra_args:
        print(f"Forwarded overrides: {extra_args}")

    try:
        subprocess.run(
            ["bash", str(script_path), str(task_file)],
            env=env,
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"Evaluation script failed with return code: {e.returncode}")
        failed_tasks = output_dir / "failed_tasks.txt"
        if failed_tasks.exists() and failed_tasks.stat().st_size > 0:
            print(f"Failed subtask list: {failed_tasks}")
            print(failed_tasks.read_text(encoding='utf-8'))
        raise


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    """LIBERO 多 GPU 评估管理器入口函数。

    使用 Hydra 配置驱动，完成以下工作:
        1. 验证必要配置（ckpt、output_dir）。
        2. 解析 Hydra task 选择。
        3. 创建任务列表文件（通过 LIBERO benchmark API 枚举任务）。
        4. 保存管理器配置副本。
        5. 如非 create_only 模式，启动并行评估。

    Args:
        cfg: Hydra 配置，需包含:
            - ckpt: 检查点路径 (必需)
            - EVALUATION.output_dir: 输出目录 (必需)
            - MULTIRUN.task_suite_names: 任务套件名称列表
            - MULTIRUN.num_gpus: GPU 数量
            - MULTIRUN.max_tasks_per_gpu: 每 GPU 最大并发任务数
            - MULTIRUN.task_file: 可选的任务列表文件路径
            - MULTIRUN.create_only: 是否仅创建任务列表
    """
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")

    task_choice = _resolve_worker_task_choice()
    manager = cfg.MULTIRUN

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)

    task_file_cfg = manager.get("task_file")
    if task_file_cfg:
        task_file = Path(os.path.expanduser(os.path.expandvars(str(task_file_cfg))))
    else:
        task_file = output_dir / "tasks.txt"
    task_file = create_task_file(task_file, list(manager.task_suite_names))

    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    if bool(manager.get("create_only", False)):
        print("create_only=True, only create the task list and exit.")
        return

    run_evaluation(
        task_file=task_file,
        task_choice=task_choice,
        ckpt=str(cfg.ckpt),
        num_gpus=int(manager.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(manager.max_tasks_per_gpu),
        output_dir=output_dir,
        extra_overrides=collect_worker_overrides(),
    )


if __name__ == "__main__":
    main()
