"""
FastWAM 项目中 Wan2.2 模型的 I/O 工具模块。

本模块提供了模型配置下载管理和权重文件加载的核心功能：
  - ModelConfig: 用于管理模型文件下载的数据类，支持从 HuggingFace 和 ModelScope
    自动下载模型文件，支持通配符文件模式匹配。
  - load_state_dict: 统一的权重加载函数，支持 .safetensors 和 .bin 格式，
    可自动合并多文件分片。
  - hash_model_file: 基于模型文件键名和形状计算哈希值，用于模型类型注册和检测。

本模块兼容 DiffSynth 框架的下载和文件管理约定。
"""

import glob
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from safetensors import safe_open


@dataclass
class ModelConfig:
    """
    模型下载和管理配置的数据类。

    封装了从 HuggingFace 或 ModelScope 下载模型的完整逻辑，包括：
      - 源模型 ID 和文件匹配模式
      - 本地缓存路径管理
      - 增量下载（跳过已下载文件）
      - 环境变量覆盖（下载源、基路径、是否跳过下载）

    Attributes:
        path: 模型文件的本地路径（下载后自动填充）。
        model_id: HuggingFace 或 ModelScope 上的模型仓库 ID。
        origin_file_pattern: 需要下载的文件通配符模式。
        download_source: 下载源（"modelscope" 或 "huggingface"），可被环境变量覆盖。
        local_model_path: 本地模型缓存根目录，可被环境变量 DIFFSYNTH_MODEL_BASE_PATH 覆盖。
        skip_download: 是否跳过下载（仅检查本地文件），可被环境变量覆盖。
        state_dict: 直接传入的状态字典（用于绕过文件加载）。
    """
    path: Union[str, list[str], None] = None
    model_id: Optional[str] = None
    origin_file_pattern: Union[str, list[str], None] = None
    download_source: Optional[str] = None
    local_model_path: Optional[str] = None
    skip_download: Optional[bool] = None
    state_dict: Optional[Dict[str, torch.Tensor]] = None

    def check_input(self):
        """检查输入参数的合法性，确保至少提供了 `path` 或 `model_id`。"""
        if self.path is None and self.model_id is None:
            raise ValueError("ModelConfig requires either `path` or (`model_id`, `origin_file_pattern`).")

    def parse_original_file_pattern(self):
        """
        解析文件匹配模式。

        - 若 pattern 为 None / "" / "./"，返回 "*" 匹配所有文件。
        - 若 pattern 以 "/" 结尾，自动追加 "*" 进行目录匹配。
        - 若 pattern 为 list，则直接返回（多文件模式）。

        Returns:
            解析后的文件匹配模式（字符串或字符串列表）。
        """
        if self.origin_file_pattern in [None, "", "./"]:
            return "*"
        if isinstance(self.origin_file_pattern, list):
            return self.origin_file_pattern
        if self.origin_file_pattern.endswith("/"):
            return self.origin_file_pattern + "*"
        return self.origin_file_pattern

    def parse_download_source(self):
        """
        解析下载源。

        优先使用实例属性 self.download_source，其次读取环境变量
        DIFFSYNTH_DOWNLOAD_SOURCE，若均未设置则默认使用 "modelscope"。

        Returns:
            下载源的字符串标识（"modelscope" 或 "huggingface"）。
        """
        if self.download_source is not None:
            return self.download_source
        env = os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE")
        return env if env is not None else "modelscope"

    def parse_skip_download(self):
        """
        解析是否跳过下载的标志。

        优先使用 self.skip_download，其次读取环境变量 DIFFSYNTH_SKIP_DOWNLOAD。
        环境变量中不区分大小写的 "true" 表示跳过下载。

        Returns:
            布尔值，True 表示跳过下载。
        """
        if self.skip_download is not None:
            return self.skip_download
        env = os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD")
        if env is None:
            return False
        return env.lower() == "true"

    def reset_local_model_path(self):
        """
        重置本地模型路径。

        优先使用环境变量 DIFFSYNTH_MODEL_BASE_PATH，
        否则若 self.local_model_path 为 None，则默认使用 "./checkpoints/"。
        """
        if os.environ.get("DIFFSYNTH_MODEL_BASE_PATH") is not None:
            self.local_model_path = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
        elif self.local_model_path is None:
            self.local_model_path = "./checkpoints/"

    def require_downloading(self):
        """
        判断是否需要下载模型文件。

        逻辑：
          1. 如果 self.path 已存在，则无需下载。
          2. 在本地缓存目录中按文件模式搜索，若找到匹配文件则无需下载。
          3. 若所有文件均未找到，且 skip_download 为 False，则需要下载。

        Returns:
            布尔值，True 表示需要从远程下载。
        """
        if self.path is not None:
            return False
        origin_file_pattern = self.parse_original_file_pattern()
        local_root = os.path.join(self.local_model_path, self.model_id)
        # 检查本地是否已存在匹配文件
        if isinstance(origin_file_pattern, list):
            # 多文件模式：所有模式都必须有匹配文件才算存在
            all_exist = True
            for pattern in origin_file_pattern:
                matches = glob.glob(os.path.join(local_root, pattern))
                if len(matches) == 0:
                    all_exist = False
                    break
            if all_exist:
                return False
        else:
            # 单文件模式：任意一个匹配文件即可
            if len(glob.glob(os.path.join(local_root, origin_file_pattern))) > 0:
                return False
        return not self.parse_skip_download()

    def download(self):
        """
        执行模型文件的远程下载。

        根据 download_source 选择下载后端：
          - "modelscope": 使用 ModelScope 的 snapshot_download
          - "huggingface": 使用 HuggingFace Hub 的 snapshot_download

        支持增量下载：通过 ignore_file_pattern 跳过已下载的文件。
        """
        origin_file_pattern = self.parse_original_file_pattern()
        root = os.path.join(self.local_model_path, self.model_id)
        # 获取当前已下载的文件列表，用于增量下载
        downloaded_files = glob.glob(origin_file_pattern, root_dir=root)
        download_source = self.parse_download_source().lower()
        if download_source == "modelscope":
            from modelscope import snapshot_download

            snapshot_download(
                self.model_id,
                local_dir=root,
                allow_file_pattern=origin_file_pattern,
                ignore_file_pattern=downloaded_files,
                local_files_only=False,
            )
        elif download_source == "huggingface":
            from huggingface_hub import snapshot_download as hf_snapshot_download

            hf_snapshot_download(
                self.model_id,
                local_dir=root,
                allow_patterns=origin_file_pattern,
                ignore_patterns=downloaded_files,
                local_files_only=False,
            )
        else:
            raise ValueError("`download_source` should be `modelscope` or `huggingface`.")

    def download_if_necessary(self):
        """
        按需下载模型文件并设置本地路径。

        完整流程：
          1. 检查输入参数合法性。
          2. 重置本地模型缓存路径。
          3. 若需要下载，则执行远程下载。
          4. 根据文件模式在本地缓存目录中查找匹配文件，填充 self.path。
          5. 如果 self.path 是单元素列表，则提取为标量路径。

        最终 self.path 可能是：
          - 单个文件路径（字符串）
          - 多个文件路径（字符串列表）
          - 目录路径（当 origin_file_pattern 为 None / "" / "./" 时）
        """
        self.check_input()
        self.reset_local_model_path()
        if self.require_downloading():
            self.download()
        # 下载完成后，自动解析本地文件路径
        if self.path is None:
            if self.origin_file_pattern in [None, "", "./"]:
                self.path = os.path.join(self.local_model_path, self.model_id)
            else:
                matches = glob.glob(os.path.join(self.local_model_path, self.model_id, self.origin_file_pattern))
                matches.sort()
                self.path = matches
        # 如果匹配到唯一文件，将列表简化为字符串
        if isinstance(self.path, list) and len(self.path) == 1:
            self.path = self.path[0]


def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    """
    加载模型权重文件的统一接口。

    支持 .safetensors 和 .bin (PyTorch) 两种格式。
    若 file_path 为列表（多文件分片），则递归加载并合并所有文件。

    Args:
        file_path: 权重文件路径（字符串）或路径列表（多文件分片）。
        torch_dtype: 可选的目标数据类型（如 torch.bfloat16）。
        device: 张量加载目标设备（默认 "cpu"）。

    Returns:
        包含所有权重的状态字典 {key: torch.Tensor}。

    示例:
        >>> # 加载单个 safetensors 文件
        >>> sd = load_state_dict("model.safetensors", torch_dtype=torch.bfloat16)
        >>> # 加载多文件分片
        >>> sd = load_state_dict(["model-00001.safetensors", "model-00002.safetensors"])
    """
    if isinstance(file_path, list):
        # 多文件分片：逐个加载并合并
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_state_dict(file_path_, torch_dtype=torch_dtype, device=device))
        return state_dict
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    """
    从 .safetensors 文件加载权重。

    使用 safetensors 库的安全加载机制（无 pickle 反序列化风险）。

    Args:
        file_path: .safetensors 文件的路径。
        torch_dtype: 目标数据类型。
        device: 张量设备（如 "cpu" / "cuda:0"）。

    Returns:
        加载完成的权重状态字典。
    """
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            value = f.get_tensor(key)
            if torch_dtype is not None:
                value = value.to(torch_dtype)
            state_dict[key] = value
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    """
    从 .bin (PyTorch pickle) 文件加载权重。

    自动处理常见的包裹键名（"state_dict"、"module"、"model_state"）。
    当状态字典只有一个顶层键且为上述包裹键时，自动解包。

    Args:
        file_path: .bin 或 .pth 文件的路径。
        torch_dtype: 目标数据类型。
        device: 张量映射设备。

    Returns:
        加载完成的权重状态字典。
    """
    # weights_only=True 提高安全性，防止 pickle 投毒
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    # 处理常见的包裹键名：若只有一项且为包裹键则自动解包
    if len(state_dict) == 1:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
    # 批量转换数据类型
    if torch_dtype is not None:
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(torch_dtype)
    return state_dict


def _load_keys_dict_from_safetensors(file_path):
    """
    从 .safetensors 文件加载键名和形状信息（不加载实际张量数据）。

    用于哈希计算，避免加载完整权重文件。

    Args:
        file_path: .safetensors 文件路径。

    Returns:
        键名字典 {key: shape_tuple}。
    """
    keys_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            keys_dict[key] = f.get_slice(key).get_shape()
    return keys_dict


def _convert_state_dict_to_keys_dict(state_dict):
    """
    将状态字典转换为仅含键名和形状的字典。

    递归处理嵌套字典结构。

    Args:
        state_dict: 原始状态字典（可能包含嵌套）。

    Returns:
        简化后的键名-形状字典。
    """
    keys_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # 记录张量的形状信息
            keys_dict[key] = list(value.shape)
        else:
            # 递归处理嵌套字典
            keys_dict[key] = _convert_state_dict_to_keys_dict(value)
    return keys_dict


def _load_keys_dict_from_bin(file_path):
    """
    从 .bin 文件加载键名和形状信息。

    先加载完整状态字典，再转换为键名-形状字典。

    Args:
        file_path: .bin 文件路径。

    Returns:
        键名-形状字典。
    """
    state_dict = load_state_dict_from_bin(file_path)
    return _convert_state_dict_to_keys_dict(state_dict)


def _load_keys_dict(file_path):
    """
    统一接口：从模型文件加载键名和形状信息。

    支持单文件和多文件分片列表。

    Args:
        file_path: 文件路径或路径列表。

    Returns:
        合并后的键名-形状字典。
    """
    if isinstance(file_path, list):
        merged = {}
        for path in file_path:
            merged.update(_load_keys_dict(path))
        return merged
    if file_path.endswith(".safetensors"):
        return _load_keys_dict_from_safetensors(file_path)
    return _load_keys_dict_from_bin(file_path)


def _convert_keys_dict_to_single_str(keys_dict, with_shape=True):
    """
    将键名-形状字典转换为逗号分隔的字符串。

    用于生成模型文件的唯一哈希标识。
    嵌套字典以 "|" 分隔，形状信息以 ":" 附加在键名后。

    Args:
        keys_dict: 键名-形状字典。
        with_shape: 是否在字符串中包含形状信息。

    Returns:
        排序后的逗号分隔字符串。
    """
    keys = []
    for key, value in keys_dict.items():
        if isinstance(key, str):
            if isinstance(value, dict):
                # 递归处理嵌套，使用 "|" 分隔层级
                keys.append(key + "|" + _convert_keys_dict_to_single_str(value, with_shape=with_shape))
            else:
                # 键名后附加形状信息，使用 ":" 分隔
                if with_shape:
                    shape = "_".join(map(str, list(value)))
                    keys.append(key + ":" + shape)
                keys.append(key)
    keys.sort()
    return ",".join(keys)


def hash_model_file(path, with_shape=True):
    """
    计算模型文件的唯一哈希值（基于键名和形状，而非文件内容）。

    哈希基于所有张量的键名和形状信息，这足以区分不同的模型架构。
    这种方式无需加载完整的权重数据，速度快且内存占用小。

    与 DiffSynth 框架的哈希计算方式兼容。

    Args:
        path: 模型文件路径（或路径列表）。
        with_shape: 计算哈希时是否包含形状信息。

    Returns:
        32 字符的 MD5 十六进制摘要。

    示例:
        >>> hash_model_file("Wan2.2_VAE.safetensors")
        "e1de6c02cdac79f8b739f4d3698cd216"
    """
    keys_dict = _load_keys_dict(path)
    keys_str = _convert_keys_dict_to_single_str(keys_dict, with_shape=with_shape).encode("UTF-8")
    return hashlib.md5(keys_str).hexdigest()
