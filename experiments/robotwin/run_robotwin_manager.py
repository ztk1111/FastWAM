"""
RoboTwin 多任务评估管理模块 (Manager)。

该模块负责在多 GPU 环境下并行调度 RoboTwin 任务的评估。
核心功能：
1. 从配置或任务列表文件中加载所有待评估任务。
2. 使用 GPU 轮询调度算法，将任务分配给空闲 GPU 执行。
3. 每个任务按 clean -> random 两个阶段顺序评估。
4. 实时监控子进程状态，收集成功率和失败信息。
5. 输出 CSV/JSON 格式的汇总结果。

调度策略：
- 每个 GPU 最多同时运行 max_tasks_per_gpu 个任务。
- 一个任务完成后，立即从 pending 队列取出下一个任务分配给该 GPU。
- 如果任一任务失败，终止所有正在运行的任务。

输出目录结构：
  evaluate_results/robotwin/<ckpt_tag>/<run_ts>/
    ├── manager.log          # 管理器的运行日志
    ├── summary.csv          # CSV 格式汇总（含每个任务和总体成功率）
    ├── summary.json         # JSON 格式详细汇总
    └── failed_tasks.txt     # 失败任务记录
"""

import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    """解析路径字符串为绝对路径。

    支持环境变量 ($HOME, $VAR) 和用户目录 (~) 展开。
    如果输入是相对路径，则相对于 base 目录解析。

    Args:
        path_str: 待解析的路径字符串。
        base: 相对路径的基准目录。

    Returns:
        Path: 解析后的绝对路径。
    """
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    """从检查点路径提取标签用于输出目录组织。

    规则:
        - 如果路径包含 "runs" 目录: 格式为 "<任务名>_<日期目录>"
        - 否则: 使用文件的 stem（不含扩展名）

    Args:
        ckpt_path: 模型检查点路径。

    Returns:
        str: 检查点标签。
    """
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    """判断一个 Hydra override 是否应被屏蔽（不传递给 worker 进程）。

    被屏蔽的 override 类型:
        - 任务级别特定参数 (ckpt, gpu_id, task_name 等)
        - MULTIRUN.* 和 hydra.* 参数（管理器专用）

    Args:
        raw_override: 原始 Hydra override 字符串。

    Returns:
        bool: 如果应屏蔽则返回 True。
    """
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    """收集所有非屏蔽的 Hydra override，传递给 worker 进程。

    从 Hydra 配置中提取所有 override，过滤掉管理器专用的参数。

    Returns:
        list[str]: 需要传递给 worker 的 override 列表。
    """
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_all_tasks() -> list[str]:
    """从 _eval_step_limit.yml 文件中加载所有可用的 RoboTwin 任务名。

    读取任务限制配置文件，提取所有任务名称，保持原始顺序并去重。

    Returns:
        list[str]: 去重后的任务名称列表。

    Raises:
        FileNotFoundError: 任务列表文件不存在时抛出。
        ValueError: 任务列表格式无效时抛出。
    """
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")
    tasks = list(task_map.keys())
    # 保持原始顺序并去重。
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    """从结果文件中解析成功率数值。

    结果文件包含每行一个浮点数，取最后一行作为最终成功率。

    Args:
        result_file: 结果文件路径。

    Returns:
        float: 解析得到的成功率。

    Raises:
        FileNotFoundError: 结果文件不存在。
        ValueError: 无法从文件中解析出有效的浮点数。
    """
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    """根据评估阶段返回对应的结果文件名。

    Args:
        phase: 评估阶段，只能是 "clean" 或 "random"。

    Returns:
        str: 对应阶段的结果文件名。

    Raises:
        ValueError: 不支持的阶段名。
    """
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _mean_or_none(values: list[float | None]) -> float | None:
    """计算一组浮点数的均值，如果列表为空则返回 None。

    自动跳过 None 值。

    Args:
        values: 可能包含 None 的浮点数列表。

    Returns:
        float | None: 均值或 None。
    """
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    """将值转换为 JSON 可序列化的格式。

    Args:
        value: 浮点数或 None。

    Returns:
        float | None: 转换后的值。
    """
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    """运行中任务的状态记录。

    Attributes:
        task_name: 任务名称。
        gpu_id: 运行的 GPU 编号。
        phase: 当前评估阶段 ("clean" 或 "random")。
        process: 子进程对象，用于监控状态和获取返回码。
    """
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    """RoboTwin 多 GPU 多任务评估管理器入口函数。

    调度策略:
        1. 加载所有待评估任务（从配置或默认任务列表）。
        2. 对每个 GPU，按 max_tasks_per_gpu 容量启动初始任务 (clean 阶段)。
        3. 轮询所有运行中任务的子进程状态。
        4. 任务完成后:
           - 如果是 clean 阶段，立即启动同任务的 random 阶段。
           - 如果是 random 阶段，从 pending 队列分配新任务给该 GPU。
        5. 任一任务失败则终止所有运行中任务。
        6. 所有任务完成后，输出 CSV 和 JSON 汇总。

    Args:
        cfg: Hydra 配置，需包含:
            - ckpt: 检查点路径 (必需)
            - MULTIRUN.num_gpus: GPU 数量 (必需)
            - MULTIRUN.max_tasks_per_gpu: 每 GPU 最大并发任务数 (必需)
            - EVALUATION.task_name: 可选，指定单个任务名
            - EVALUATION.output_dir: 输出目录
    """
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    gpu_ids = list(range(num_gpus))

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = _load_all_tasks()
    else:
        tasks = [str(task_name_cfg)]

    extra_overrides = _collect_worker_overrides()

    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []

    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }

    def log(msg: str) -> None:
        """记录一条带时间戳的日志，同时打印到终端和写入日志文件。"""
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        """构建调用 eval_robotwin_single.py 的命令。

        Args:
            task_name: 任务名称。
            gpu_id: GPU 编号。
            phase: 评估阶段 ("clean" | "random")。

        Returns:
            list[str]: 完整的命令行列表。
        """
        task_config = phase_to_task_config[phase]
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(output_dir)}",
        ]
        cmd.extend(extra_overrides)
        return cmd

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        """启动一个任务阶段评估子进程。

        Args:
            task_name: 任务名称。
            gpu_id: GPU 编号。
            phase: 评估阶段 ("clean" | "random")。

        Returns:
            RunningState: 记录运行状态的数据类实例。
        """
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} "
            f"cmd={' '.join(cmd)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
        )

    def terminate_all_running() -> None:
        """优雅地终止所有正在运行的任务子进程。

        先尝试 SIGTERM，等待超时后强制 SIGKILL。
        """
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()
        # 等待所有进程结束，超时则强制杀死
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        """统计指定 GPU 上当前正在运行的任务数。

        Args:
            gpu_id: GPU 编号。

        Returns:
            int: 正在运行的任务数量。
        """
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def try_launch_pending(gpu_id: int) -> None:
        """尝试为指定 GPU 从 pending 队列启动新的任务。

        在 GPU 容量允许的情况下，从队列中取出任务并以 clean 阶段启动。

        Args:
            gpu_id: GPU 编号。
        """
        while len(pending_tasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name = pending_tasks.popleft()
            running_states.append(launch_phase(task_name=task_name, gpu_id=gpu_id, phase="clean"))

    def write_outputs() -> None:
        """将汇总结果写入 CSV 和 JSON 文件，同时记录失败任务。"""
        clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
        random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow(
                    [
                        task,
                        task_rates[task]["clean"],
                        task_rates[task]["random"],
                    ]
                )
            writer.writerow(["__overall__", clean_mean, random_mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                    "random_success_rate": _to_jsonable(task_rates[task]["random"]),
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} output_dir={run_output_dir}"
    )

    # 初始启动：为每个 GPU 分配初始任务（clean 阶段）
    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)

    has_failure = False
    failure_message = ""

    # 主调度循环：轮询所有运行中任务的状态
    while len(running_states) > 0:
        progressed = False  # 本轮循环是否有任务完成或出错
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue  # 任务仍在运行
            progressed = True
            running_states.remove(state)

            # 情况 1: 任务进程失败
            if return_code != 0:
                has_failure = True
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, return_code={return_code}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            # 情况 2: 任务成功完成，解析成功率文件
            result_file = run_output_dir / state.task_name / _phase_result_filename(state.phase)
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, error={repr(exc)}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            task_rates[state.task_name][state.phase] = success_rate
            log(
                f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                f"success_rate={success_rate:.4f}"
            )

            # clean 阶段完成 -> 立即启动同任务的 random 阶段
            if state.phase == "clean":
                running_states.append(launch_phase(
                    task_name=state.task_name,
                    gpu_id=gpu_id,
                    phase="random",
                ))
                continue

            # random 阶段完成 -> 从 pending 队列分配新任务
            try_launch_pending(gpu_id)

        # 发生失败时退出主循环
        if has_failure:
            break
        # 本轮无任务完成，等待后再次轮询
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    # 如果发生失败，记录尚未启动的任务
    if has_failure:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "phase": "not_started",
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")


if __name__ == "__main__":
    main()
