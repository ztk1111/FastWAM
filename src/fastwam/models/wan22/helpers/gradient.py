"""
FastWAM 项目的梯度检查点（Gradient Checkpointing）工具模块。

该模块提供了梯度检查点功能的封装函数，用于在训练时以计算换内存：
在前向传播时丢弃中间激活值，在反向传播时重新计算，从而大幅降低 GPU 显存占用。

本模块是 torch.utils.checkpoint.checkpoint 的轻量封装，
增加了对非重入模式（use_reentrant=False）的默认支持，兼容 Transformer 类模型。
"""

import torch


def create_custom_forward(module):
    """
    为梯度检查点创建自定义前向函数包装器。

    torch.utils.checkpoint.checkpoint 要求传入的函数不接收关键字参数，
    但很多 PyTorch 模块的前向函数可能包含 kwargs。此包装器将所有输入
    （包括 kwargs）直接传递给模块的 __call__ 方法。

    Args:
        module: 需要包装的 PyTorch 模块。

    Returns:
        一个仅接收位置参数的 custom_forward 函数，内部将参数全部转发给 module。

    示例:
        >>> model = WanVideoDiT(...)
        >>> wrapped = create_custom_forward(model)
        >>> output = wrapped(x, timestep, context)  # 效果等价于 model(x, timestep, context)
    """
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    *args,
    **kwargs,
):
    """
    梯度检查点的条件前向传播函数。

    根据 use_gradient_checkpointing 标志决定是否启用梯度检查点。
    启用时，使用 torch.utils.checkpoint.checkpoint 包装前向传播，
    训练时中间激活值不会被保存，反向传播时重新计算。

    参数说明:
        model: 进行前向传播的 PyTorch 模型。
        use_gradient_checkpointing: 布尔标志，是否启用梯度检查点。
        *args: 传递给模型的位置参数。
        **kwargs: 传递给模型的关键字参数。

    返回:
        模型前向传播的输出，与直接在 model(*args, **kwargs) 上调用结果一致。

    示例:
        >>> # 启用梯度检查点
        >>> output = gradient_checkpoint_forward(dit, True, x, t, context)
        >>> # 禁用（直接前向，节省计算时间）
        >>> output = gradient_checkpoint_forward(dit, False, x, t, context)

    注意:
        使用 use_reentrant=False 模式，该模式在 PyTorch 2.x 中更稳定，
        且兼容 nn.Dropout 等模块。
    """
    if use_gradient_checkpointing:
        # 启用梯度检查点：在前向传播中不保存中间激活值
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,  # 使用非重入模式，更安全稳定
        )
    else:
        # 不使用梯度检查点，直接前向传播
        model_output = model(*args, **kwargs)
    return model_output
