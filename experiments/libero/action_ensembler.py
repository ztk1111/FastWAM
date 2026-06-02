"""
动作集成器 (ActionEnsembler) 工具模块。

在重规划（re-plan）控制策略中，同一时间步可能被多个动作块覆盖，
该模块通过对同一时间步的多个预测动作取平均，实现平滑的动作集成。

原理示意:
    - 第一次推理: 预测动作 A0, A1, A2, A3, A4  (action_horizon=5)
    - 执行 A0, A1 后重规划 (replan_steps=2)
    - 第二次推理: 预测动作 B2, B3, B4, B5, B6
    - 在时间步 2, 3, 4 上，A 和 B 的预测被平均: (A2+B2)/2, (A3+B3)/2, (A4+B4)/2
    - 这种集成可以平滑动作过渡，减少抖动
"""

from collections import defaultdict
import numpy as np
import torch

class ActionEnsembler:
    """动作集成器：通过多步预测平均实现平滑动作。

    在重规划控制中，将多次推理对同一时间步的预测动作取平均，
    减少单次推理的噪声和抖动，提高执行稳定性。

    使用示例:
        >>> ensembler = ActionEnsembler()
        >>> action_chunk = np.random.randn(5, 7)  # [T=5, D=7]
        >>> ensembler.add_actions(action_chunk, start_timestamp=0)
        >>> # 在时间步 0 获取集成后的动作
        >>> action = ensembler.get_action(timestamp=0)
        >>> action.shape
        (7,)
    """

    def __init__(self):
        """初始化动作集成器。

        action_cache: 字典，key=时间戳，value=该时间步的所有预测动作列表。
        """
        self.action_cache = defaultdict(list)

    def reset(self):
        """清空所有缓存的动作，开始新的 episode。"""
        self.action_cache.clear()

    def add_actions(self, action_chunk: np.ndarray, start_timestamp: int):
        """添加一次推理生成的动作块到缓存。

        将 action_chunk 中的每个动作按时间戳索引存储。
        如果多个推理覆盖同一时间戳，后面的预测会追加到列表中。

        Args:
            action_chunk: 动作块数组，形状 [T, D] 或 [1, T, D]。
            start_timestamp: 该动作块的起始时间戳（当前仿真步数）。

        示例:
            >>> chunk = np.array([[0.1, 0.2], [0.3, 0.4]])  # T=2, D=2
            >>> ensembler.add_actions(chunk, start_timestamp=5)
            # 存储: cache[5] = [[0.1, 0.2]], cache[6] = [[0.3, 0.4]]
        """
        if action_chunk.ndim == 3:
            # 移除 batch 维度（来自模型输出 [1, T, D]）
            action_chunk = action_chunk.squeeze(0)
        horizon, action_dim = action_chunk.shape

        for i in range(horizon):
            target_ts = start_timestamp + i
            self.action_cache[target_ts].append(action_chunk[i, :])

    def get_action(self, timestamp: int) -> np.ndarray:
        """获取指定时间步的集成后动作。

        对该时间步所有缓存的预测进行平均，得到平滑后的动作。

        Args:
            timestamp: 要获取动作的时间步。

        Returns:
            np.ndarray: 平均后的动作向量，形状 [D]。

        Raises:
            ValueError: 如果指定时间步没有任何缓存动作。

        示例:
            >>> action = ensembler.get_action(0)
            >>> action  # 所有预测在时间步 0 的平均值
        """
        if timestamp not in self.action_cache:
            raise ValueError(f"No actions cached for timestamp {timestamp}")
        preds = self.action_cache[timestamp]
        stacked_preds = np.stack(preds, axis=0)
        # 对所有预测求平均作为集成动作
        averaged_action = np.mean(stacked_preds, axis=0)
        return averaged_action

    def _cleanup(self, current_timestamp: int):
        """清理已过期的时间步缓存，释放内存。

        删除所有早于 current_timestamp 的缓存条目，
        这些时间步的动作已经执行完毕，不再需要。

        Args:
            current_timestamp: 当前时间步，更早的将被清理。
        """
        keys_to_delete = [ts for ts in self.action_cache.keys() if ts < current_timestamp]
        for ts in keys_to_delete:
            del self.action_cache[ts]