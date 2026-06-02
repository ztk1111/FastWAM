"""
FastWAM 项目中 Wan2.2 模型的连续时间 Flow Matching 调度器模块。

该模块实现了基于 Flow Matching (FM) 框架的连续时间调度器，
用于扩散模型的训练和推理调度。核心技术要点：

1. Flow Matching 概览
   Flow Matching 使用线性插值路径连接数据分布 x0 和噪声分布 x1：
     xt = (1 - sigma) * x0 + sigma * noise
   其中 sigma 是时间步 t 归一化到 [0, 1] 的结果。

2. 训练阶段
   - sample_training_t: 从 [0, T] 均匀采样时间步，经 shift 函数重分布。
   - add_noise: 根据时间步对原始数据进行加噪，构造训练样本。
   - training_target: 计算 velocity 目标值 (noise - sample)。
   - training_weight: 计算每个时间步的损失权重（中心高斯加权）。

3. 推理阶段
   - build_inference_schedule: 构建离散推理步进计划和 delta 步长。
   - step: 执行单步去噪更新（欧拉法）。

4. Shift 机制
   通过非线性映射 _phi 调整时间步分布，使得更多的采样点集中在
   靠近原始数据的区域，提高感知质量。

参考文献:
    - Flow Matching: https://arxiv.org/abs/2210.02747
    - Wan2.1 Technical Report: https://arxiv.org/abs/2501.15836
"""

import torch


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling.

    Wan2.2 模型使用的连续时间 Flow Matching 调度器，支持 shift 采样重分布。
    提供训练所需的加噪、目标计算、权重计算功能，以及推理所需的调度构建和步进功能。
    """

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        """
        初始化 Flow Matching 调度器。

        Args:
            num_train_timesteps: 训练时间步数（离散化精细度，默认 1000）。
            shift: 时间步重分布参数（默认 5.0）。
                   shift 越大，采样点越集中在靠近数据的一侧。
            eps: 权重计算时的数值稳定性常数，防止除零。
        """
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        # 预计算训练权重的统计量（y_min 和归一化常数）
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        """
        非线性时间步映射函数（Shift 变换）。

        将均匀分布的 u in [0, 1] 映射到非均匀分布的 sigma in [0, 1]。
        通过可调节的 shift 参数控制重分布强度：
          - shift = 1.0: sigma = u（均匀映射）
          - shift > 1.0: 更多采样点集中在靠近 0 的区域（靠近原始数据）
          - shift 越大，重分布效果越强

        映射公式:
            sigma = shift * u / (1.0 + (shift - 1.0) * u)

        Args:
            u: 输入张量，范围 [0, 1]。
            shift: 控制参数，需为正数。

        Returns:
            映射后的 sigma 值，与 u 形状相同。

        示例:
            >>> u = torch.tensor([0.0, 0.5, 1.0])
            >>> WanContinuousFlowMatchScheduler._phi(u, shift=5.0)
            tensor([0.0000, 0.8333, 1.0000])  # 中间值被推向 1
        """
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        """
        预计算训练权重的高斯统计量。

        在初始化时提前计算权重函数所需的偏移量和归一化常数，避免训练中重复计算。

        计算流程：
          1. 将 [0, T] 均匀网格通过 _phi 映射到非均匀时间步网格。
          2. 以 T/2 为中心计算高斯权重 y = exp(-2 * ((t - T/2) / T)^2)。
          3. 记录 y 的最小值 y_min，用于平移。
          4. 计算平移后的均值作为归一化常数。

        Returns:
            (y_min, norm_const) 元组：
              - y_min: 高斯权重的最小值（偏移量）。
              - norm_const: 平移后权重的均值（归一化常数）。
        """
        steps = self.num_train_timesteps
        # 在 [1.0, 0.0] 范围内均匀采样（反向，从 1 到 0）
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        # 应用 shift 映射并缩放到 [0, T]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        # 以 T/2 为中心的高斯权重
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        # 平移后计算均值作为归一化常数
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        随机采样训练时间步。

        从 [0, T] 范围内采样 batch_size 个时间步，用于训练时构造
        带噪样本。采样分布受 shift 参数控制，非均匀分布在时间轴上。

        Args:
            batch_size: 批量大小。
            device: 目标设备（如 torch.device("cuda")）。
            dtype: 目标数据类型（如 torch.bfloat16）。

        Returns:
            形状为 (batch_size,) 的时间步张量，取值范围 [0, T]。

        示例:
            >>> scheduler = WanContinuousFlowMatchScheduler()
            >>> t = scheduler.sample_training_t(4, torch.device("cuda"), torch.float32)
            >>> t.shape  # (4,)，值为 [0.0, 1000.0) 区间
        """
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        # 从均匀分布 [0, 1) 采样
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        # 通过 shift 函数重分布
        sigma = self._phi(u, self.shift)
        # 缩放到 [0, T] 区间
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        计算训练时每个时间步的损失权重。

        采用中心高斯加权策略：对靠近中间时间步的样本赋予更高权重，
        以调节模型在不同噪声水平的关注度。

        权重公式:
            y = exp(-2 * ((t - T/2) / T)^2)
            weight = (y - y_min) / norm_const

        Args:
            timestep: 时间步张量，形状 (batch_size,) 或标量。
                      取值范围 [0, T]。

        Returns:
            权重张量，与 timestep 形状相同。
            标量输入返回 0 维张量，批输入返回 1 维张量。

        示例:
            >>> scheduler = WanContinuousFlowMatchScheduler()
            >>> t = torch.tensor([0.0, 500.0, 1000.0])
            >>> w = scheduler.training_weight(t)
            >>> w.shape  # (3,)
        """
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        # 中心高斯：以 T/2 为均值
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        # 平移确保非负
        y_shifted = y - self._y_min
        # 归一化
        weight = y_shifted / (self._weight_norm_const + self.eps)
        # 将标量情况恢复为 0 维张量
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        根据 Flow Matching 的线性插值路径对原始样本加噪。

        加噪公式:
            xt = (1 - sigma) * x0 + sigma * noise
        其中 sigma = t / T 是归一化的时间步。

        Args:
            original_samples: 原始数据样本。
                形状: (batch_size, C, T, H, W) 或 (batch_size, C, H, W)。
            noise: 噪声张量，形状与 original_samples 相同。
            timestep: 时间步，形状 (batch_size,) 或标量。

        Returns:
            加噪后的样本，形状与 original_samples 相同。

        示例:
            >>> scheduler = WanContinuousFlowMatchScheduler()
            >>> x0 = torch.randn(2, 3, 16, 64, 64)  # (batch, ch, frames, h, w)
            >>> noise = torch.randn_like(x0)
            >>> t = torch.tensor([300.0, 700.0])
            >>> xt = scheduler.add_noise(x0, noise, t)
            >>> xt.shape  # (2, 3, 16, 64, 64)
        """
        # sigma = t / T，取值范围 [0, 1]
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            # 标量情况：直接广播
            return (1 - sigma) * original_samples + sigma * noise
        # 批量情况：将 sigma 的形状从 (B,) 扩展到可广播的维度
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        计算 Flow Matching 训练目标（velocity 场）。

        对于线性插值路径 xt = (1-sigma)*x0 + sigma*noise，
        velocity 目标为 d(xt)/dsigma = noise - x0。

        Args:
            sample: 加噪后的样本 xt（未使用，保留仅用于 API 兼容）。
                形状: (batch_size, ...)。
            noise: 原始噪声张量。
                形状: (batch_size, ...)。
            timestep: 时间步（未使用，保留仅用于 API 兼容）。

        Returns:
            velocity 目标张量，与 sample 形状相同: noise - sample。
            代表模型需要预测的速度场方向。

        示例:
            >>> scheduler = WanContinuousFlowMatchScheduler()
            >>> xt = torch.randn(2, 3, 16, 64, 64)  # 加噪样本
            >>> noise = torch.randn_like(xt)          # 原始噪声
            >>> t = torch.tensor([500.0, 500.0])
            >>> target = scheduler.training_target(xt, noise, t)
            >>> target.shape  # (2, 3, 16, 64, 64)
        """
        del timestep  # Flow Matching 的 velocity 目标与时间步无关
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        构建推理阶段的离散化调度计划。

        将连续时间 [0, T] 离散化为 num_inference_steps 步，
        返回每个推理步骤的时间步 t 和步长 delta。

        调度流程:
          1. 在 [1.0, 0.0] 上均匀采样 num_inference_steps + 1 个点。
          2. 通过 _phi 映射得到 sigma 网格。
          3. 将 sigma 缩放到时间步 t = sigma * T。
          4. delta = sigma[i+1] - sigma[i] 作为欧拉步长。

        Args:
            num_inference_steps: 推理步数（离散化步数）。
            device: 目标设备。
            dtype: 目标数据类型。
            shift_override: 可选的 shift 覆盖值，若为 None 则使用训练时的 shift。

        Returns:
            (timesteps, deltas) 元组:
              - timesteps: 形状 (num_inference_steps,)，
                          每个推理步骤对应的时间步（在 [0, T] 范围内）。
              - deltas: 形状 (num_inference_steps,)，
                        每个推理步骤对应的 sigma 步长（用于欧拉更新）。

        示例:
            >>> scheduler = WanContinuousFlowMatchScheduler()
            >>> t, d = scheduler.build_inference_schedule(50, torch.device("cuda"), torch.float32)
            >>> t.shape  # (50,)
            >>> d.shape  # (50,)
            >>> t[0] > t[-1]  # 时间步从大到小（从噪声到数据）
        """
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        # 在 [1.0, 0.0] 上均匀采样（从噪声到数据方向）
        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        # 应用 shift 映射
        sigma_steps = self._phi(u_steps, shift)
        # 取前 N 个点作为时间步（不含 sigma=0 的终点）
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        # 连续两步之间的 sigma 差值作为欧拉步长
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """
        执行单步推理去噪（欧拉法）。

        根据 Flow Matching 的 ODE 更新公式:
            x_{t+1} = x_t + v(x_t, t) * delta_sigma
        其中 v 是模型预测的 velocity，delta_sigma 是 sigma 域步长。

        Args:
            model_output: 模型预测的 velocity 场。
                形状: (batch_size, C, T, H, W) 或 (batch_size, C, H, W)
            delta: sigma 步长。
                形状: 标量 (0维) 或 (batch_size,)
            sample: 当前去噪样本。
                形状: (batch_size, C, T, H, W) 或 (batch_size, C, H, W)

        Returns:
            更新后的样本，形状与 sample 相同。

        示例:
            >>> scheduler = WanContinuousFlowMatchScheduler()
            >>> xt = torch.randn(2, 3, 16, 64, 64)  # 当前样本
            >>> v = torch.randn_like(xt)             # 模型预测的 velocity
            >>> delta = torch.tensor(0.02)           # 步长
            >>> xt_next = scheduler.step(v, delta, xt)
            >>> xt_next.shape  # (2, 3, 16, 64, 64)
        """
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            # 标量步长：直接广播
            return sample + model_output * delta
        # 批量步长：将 delta 从 (B,) 扩展到可广播维度
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta
