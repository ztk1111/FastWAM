"""
RobotWin single-task evaluation entrypoint (Hydra).

Features:
- Read `configs/sim_robotwin.yaml`.
- Check or create the symlink:
  `RoboTwin/policy/fastwam -> experiments/robotwin/fastwam`.
- Forward config overrides to the official RoboTwin entrypoint
  `script/eval_policy.py` and save logs.

Common arguments:
- `ckpt`: path to the FastWAM checkpoint (required).
- `EVALUATION.task_name`: task name to evaluate (required).
- `gpu_id`: sets `CUDA_VISIBLE_DEVICES`.

Examples:
1) Minimal run
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock

2) Run with more evaluation overrides
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock \
     EVALUATION.task_config=demo_randomized \
     EVALUATION.replan_steps=4 \
     EVALUATION.num_inference_steps=4 \
     gpu_id=0

RoboTwin 单任务评估入口。

该模块负责:
1. 从 Hydra 配置解析评估参数（检查点路径、任务名称、GPU 等）。
2. 检查并创建策略源码的符号链接，使 RoboTwin 框架能找到 fastwam 策略。
3. 将配置参数转发给 RoboTwin 官方评估入口 script/eval_policy.py。
4. 捕获子进程输出并保存日志，同时保存评估配置副本。
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "fastwam_policy"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    """解析路径字符串为绝对路径。

    支持环境变量（$HOME）和用户目录（~）展开。
    若输入为相对路径，则相对于 base 目录解析。

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


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    """解析可选路径，空值或 "none"/"null" 返回 None。

    Args:
        path_value: 可选的路径值。
        base: 相对路径的基准目录。

    Returns:
        Path | None: 解析后的路径或 None。
    """
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    """解析数据集统计文件 (dataset_stats.json) 的路径。

    搜索顺序:
        1. 配置中显式指定的 EVALUATION.dataset_stats_path。
        2. 检查点路径的父目录（向上最多 4 层）。

    Args:
        cfg: Hydra 配置对象。
        ckpt_path: 模型检查点路径。

    Returns:
        Path: 找到的 dataset_stats.json 路径。

    Raises:
        FileNotFoundError: 在所有候选位置都未找到时抛出。
    """
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    # 在检查点的父目录中搜索 dataset_stats.json
    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    """从检查点路径中提取标签，用于组织输出目录。

    如果路径包含 "runs" 目录，则使用 "<任务名>_<日期目录>" 格式，
    否则使用文件名的 stem 部分（不含扩展名）。

    Args:
        ckpt_path: 模型检查点路径。

    Returns:
        str: 检查点标签字符串。

    示例:
        >>> _resolve_ckpt_tag(Path("/path/to/runs/my_task/20250301_120000/model.pt"))
        'my_task_20250301_120000'
        >>> _resolve_ckpt_tag(Path("/other/path/model.pt"))
        'model'
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


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    """确保 RoboTwin 框架的策略符号链接指向 FastWAM 策略源码。

    在 RoboTwin 的 policy 目录下创建符号链接:
        RoboTwin/policy/fastwam -> experiments/robotwin/fastwam_policy

    如果符号链接已存在，检查其目标是否匹配；如果匹配则直接返回。

    Args:
        robotwin_root: RoboTwin 项目根目录。
        policy_source_dir: FastWAM 策略源码目录。

    Returns:
        Path: 策略符号链接的路径。

    Raises:
        FileNotFoundError: RoboTwin policy 目录不存在时抛出。
        RuntimeError: 符号链接冲突或目标路径不是符号链接时抛出。
    """
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")

    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()

    # 链接不存在 -> 创建
    if not policy_target.exists() and not policy_target.is_symlink():
        policy_target.symlink_to(source_resolved, target_is_directory=True)
        return policy_target

    # 链接已存在 -> 检查一致性
    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
                f"expected -> {source_resolved}"
            )
        return policy_target

    # 已存在且不是符号链接 -> 错误
    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _format_override_value(value: Any) -> str:
    """将值格式化为 RoboTwin 命令行参数格式。

    处理 bool/None/数字/字符串等类型的正确序列化。

    Args:
        value: 待格式化的值。

    Returns:
        str: 格式化后的字符串。
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    """向 RoboTwin 命令行参数列表添加一个 --key value 对。

    Args:
        overrides: 参数列表（会被原地修改）。
        key: 参数名（不含前缀 "--"）。
        value: 参数值。
        skip_none: 如果为 True 且 value 为 None 则跳过添加。
    """
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    """RoboTwin 单任务评估入口函数。

    通过 Hydra 接收配置，完成以下工作:
        1. 验证并解析检查点路径和任务名称。
        2. 创建策略符号链接。
        3. 构造输出目录结构: evaluate_results/robotwin/<ckpt_tag>/<run_ts>/<task_name>/。
        4. 解析数据集统计文件路径。
        5. 将所有配置参数转换为 RoboTwin 命令行的 --key value 格式。
        6. 以子进程方式调用 RoboTwin 的 script/eval_policy.py 进行评估。
        7. 实时捕获子进程输出并写入日志文件。
        8. 保存评估配置副本。

    Args:
        cfg: Hydra 组合配置，需包含:
            - ckpt: 检查点路径（必需）
            - EVALUATION.task_name: 任务名称（必需）
            - EVALUATION.robotwin_root: RoboTwin 根目录
            - EVALUATION.output_dir: 输出目录
            - gpu_id: GPU 设备编号
            - 其他评估参数（可选）
    """
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    policy_source_dir = (PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")

    _ensure_policy_symlink(robotwin_root=robotwin_root, policy_source_dir=policy_source_dir)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / ckpt_tag
        / run_ts
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / (
        f"eval_{str(cfg.EVALUATION.task_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    # 构造 RoboTwin 评估输出子目录: evaluate_results/robotwin/<ckpt_tag>/<run_ts>/<task_name>/
    robotwin_eval_base = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / ckpt_tag
        / run_ts
        / str(cfg.EVALUATION.task_name)
    )

    # 获取仿真配置路径和当前任务选择
    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")

    # 解析数据集统计文件路径
    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    # 将 FastWAM 配置参数转换为 RoboTwin 的 --key value 格式
    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", cfg.EVALUATION.policy_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "eval_num_episodes", cfg.EVALUATION.eval_num_episodes)

    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
    _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
    _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
    _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
    _append_override(overrides, "rand_device", cfg.EVALUATION.rand_device)
    _append_override(overrides, "tiled", cfg.EVALUATION.tiled)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    _append_override(
        overrides,
        "skip_get_obs_within_replan",
        cfg.EVALUATION.skip_get_obs_within_replan,
    )

    # 构造 RoboTwin 评估命令
    cmd = [
        sys.executable,
        "-u",                          # 无缓冲输出
        "script/eval_policy.py",       # RoboTwin 官方评估入口
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]

    # 设置环境变量：指定 GPU 和强制 Python 无缓冲输出
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    # 启动子进程并实时捕获输出（同时写入日志文件和打印到终端）
    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    # 检查子进程返回码
    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")

    # 评估成功，保存配置副本以备追溯
    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    OmegaConf.save(
        config=cfg,
        f=str(run_output_dir / f"eval_config_{str(cfg.EVALUATION.task_name)}.yaml"),
    )


if __name__ == "__main__":
    main()
