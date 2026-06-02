"""
LIBERO 评估工具函数模块。

该模块提供在 LIBERO 仿真环境中评估策略时所需的通用工具函数，包括：
1. 环境创建：get_libero_env - 初始化 LIBERO 环境并获取任务描述。
2. 观测处理：get_libero_image - 提取并预处理仿真图像（180度旋转）。
3. 视频保存：save_rollout_video / save_prediction_video - 保存 MP4 回放视频。
4. 动作处理：get_libero_dummy_action / invert_gripper_action / binarize_gripper_open。
5. 坐标变换：quat2axisangle - 四元数转轴角表示。

数据预处理注意事项:
    - LIBERO 渲染的图像需要旋转 180 度以匹配训练时的预处理。
    - Gripper 动作符号因数据集对齐需要翻转（0=close, 1=open -> -1=open, +1=close）。

Utils for evaluating policies in LIBERO simulation environments.
"""

import math
import time
import pathlib

import imageio
from PIL import Image, ImageDraw
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from fastwam.utils.video_io import save_mp4

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def get_libero_env(task, resolution, seed, env_num=1):
    """初始化并返回 LIBERO 环境及任务描述。

    根据任务对象解析 BDDL 文件路径，配置摄像头分辨率，创建仿真环境。
    支持单环境或向量化多环境（env_num > 1）。

    Args:
        task: LIBERO 任务对象，包含 language 描述和 BDDL 文件信息。
        resolution: 摄像头渲染分辨率（宽和高相同）。
        seed: 随机种子（影响对象位置）。
        env_num: 环境数量，>1 时使用 SubprocVectorEnv。

    Returns:
        tuple: (env, task_description)
            - env: LIBERO 仿真环境实例。
            - task_description: 自然语言任务描述字符串。

    Initializes and returns the LIBERO environment, along with the task description.
    """
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if env_num > 1:
        env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # 注意: seed 会影响物体位置，即使使用固定初始状态
    return env, task_description

def get_libero_dummy_action():
    """获取无操作 (no-op) 动作，用于在机器人不执行操作时推进仿真。

    7 维动作向量: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    其中 gripper=-1 表示夹爪保持闭合。

    Returns:
        list[float]: 无操作动作 [0, 0, 0, 0, 0, 0, -1]。

    Get dummy/no-op action, used to roll out the simulation while the robot does nothing.
    """
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_image(obs):
    """从观测中提取图像并进行预处理。

    对 agentview（主摄像头）和 wrist（腕部摄像头）图像进行 180 度旋转，
    以匹配训练数据预处理步骤。

    Args:
        obs: LIBERO 环境观测字典，需包含 "agentview_image" 和
             "robot0_eye_in_hand_image" 键。

    Returns:
        dict: 包含以下键的字典:
            - "image": 处理后的主摄像头图像，ndarray 形状 [H, W, 3]。
            - "wrist_image": 处理后的腕部摄像头图像，ndarray 形状 [H, W, 3]。

    Extracts image from observations and preprocesses it.
    """
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    # 重要: 旋转 180 度以匹配训练预处理

    # [yc] 腕部摄像头图像
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    # 重要: 旋转 180 度以匹配训练预处理

    return {
        "image": img,
        "wrist_image": wrist_img
    }

def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description, log_file=None, fps=24):
    """保存 episode 的回放视频为 MP4 文件。

    支持多种图像格式输入:
        - dict: 多摄像头图像水平拼接，并标注摄像头名称。
        - PIL Image: 直接转换为 RGB 帧。
        - ndarray: 直接作为帧写入。

    Args:
        rollout_dir: 视频保存目录。
        rollout_images: 回放帧列表，每帧可以是 dict、PIL Image 或 ndarray。
        idx: episode 编号。
        success: 任务是否成功。
        task_description: 任务描述，用于文件名。
        log_file: 可选的日志文件句柄。
        fps: 视频帧率，默认 24。

    Returns:
        str: 保存的 MP4 文件路径。

    Saves an MP4 replay of an episode.
    """
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        if isinstance(img, dict):
            # 多摄像头图像：水平拼接并添加名称标注
            image = []
            for key, value in img.items():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                pil_img = Image.fromarray(value_array)
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 10), f"{key}", fill=(255, 255, 255))
                image.append(np.array(pil_img))
            frame = np.concatenate(image, axis=1)
        elif isinstance(img, Image.Image):
            frame = np.array(img.convert("RGB"))
        else:
            frame = np.array(img)
        video_writer.append_data(frame)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_prediction_video(
    rollout_dir,
    gt_frames,
    pred_frames,
    idx,
    replan_idx,
    success,
    task_description,
    log_file=None,
    fps=8,
):
    """保存未来帧预测的对比 MP4 视频。

    将真实帧（GT）和预测帧（Pred）上下拼接后保存，并标注 "gt" 和 "pred" 文字，
    便于直观对比预测质量。

    Args:
        rollout_dir: 视频保存目录。
        gt_frames: 真实帧列表（来自仿真环境）。
        pred_frames: 预测帧列表（来自模型）。
        idx: episode 编号。
        replan_idx: 重规划索引（第几次 re-plan）。
        success: 任务是否成功。
        task_description: 任务描述。
        log_file: 可选的日志文件句柄。
        fps: 视频帧率，默认 8（较慢速播放以便观察）。

    Returns:
        str: 保存的 MP4 文件路径。

    Raises:
        ValueError: 输入帧列表为空时抛出。

    Saves an MP4 comparison of ground-truth and predicted future frames for one replanning clip.
    """
    num_frames = min(len(gt_frames), len(pred_frames))
    if num_frames <= 0:
        raise ValueError("Cannot save prediction video with empty GT/pred frame lists.")

    stitched_frames = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        # 处理真实帧
        if isinstance(gt_frame, dict):
            gt_images = []
            for value in gt_frame.values():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                gt_images.append(value_array)
            gt_image = np.concatenate(gt_images, axis=1)
        elif isinstance(gt_frame, Image.Image):
            gt_image = np.array(gt_frame.convert("RGB"))
        else:
            gt_image = np.array(gt_frame)

        # 处理预测帧
        if isinstance(pred_frame, Image.Image):
            pred_image = np.array(pred_frame.convert("RGB"))
        else:
            pred_image = np.array(pred_frame)

        # 统一尺寸（以预测帧为基准）
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        # 添加标注文字并上下拼接
        gt_pil = Image.fromarray(gt_image)
        ImageDraw.Draw(gt_pil).text((10, 10), "gt", fill=(255, 255, 255))
        pred_pil = Image.fromarray(pred_image)
        ImageDraw.Draw(pred_pil).text((10, 10), "pred", fill=(255, 255, 255))
        stitched_frames.append(
            Image.fromarray(np.concatenate([np.array(pred_pil), np.array(gt_pil)], axis=0))
        )

    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    try:
        replan_tag = f"{int(replan_idx):04d}"
    except (TypeError, ValueError):
        replan_tag = str(replan_idx)
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}"
        f"--task={processed_task_description}--replan={replan_tag}--gt-pred.mp4"
    )
    save_mp4(stitched_frames, mp4_path, fps=fps)
    print(f"Saved predicted future comparison MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved predicted future comparison MP4 at path {mp4_path}\n")
    return mp4_path

def binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    """将夹爪开合值二值化。

    阈值 0.5: >0.5 表示打开，<=0.5 表示闭合。

    Args:
        open_val: 夹爪开合值（浮点数或数组）。

    Returns:
        np.ndarray: 二值化后的夹爪状态（0.0 或 1.0）。
    """
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = (v > 0.5)
    return np.asarray(bin_val, dtype=np.float32)


def quat2axisangle(quat):
    """将四元数转换为轴角表示。

    代码源自 robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    轴角表示用一个三维向量表示旋转，方向指向旋转轴，模长为旋转角度（弧度）。
    这是机器人控制中常用的旋转表示法，比四元数更适合作为神经网络输入。

    Args:
        quat (np.array): 四元数 (x, y, z, w) 浮点数数组。

    Returns:
        np.array: 轴角表示 (ax, ay, az)，单位为弧度。

    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # 限制四元数 w 分量在 [-1, 1] 范围内
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # 接近零旋转（角度=0），直接返回零向量
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def invert_gripper_action(action):
    """翻转夹爪动作的符号（动作向量的最后一维）。

    由于 RLDS 数据加载器将夹爪动作对齐为 0=闭合, 1=打开，
    但 LIBERO 仿真环境使用 -1=打开, +1=闭合 的约定，
    因此需要在执行前翻转符号。

    Args:
        action: 动作数组，形状 [..., D]，最后一维为夹爪控制。

    Returns:
        action: 符号翻转后的动作数组（原位修改并返回）。

    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
