"""
FastWAM 训练入口脚本。

该脚本是 FastWAM 项目的顶层训练入口，通过 Hydra 框架管理配置。
用户可以通过命令行覆盖配置中的任意参数（如 data.batch_size=8）。

用法:
    python scripts/train.py                          # 使用默认配置
    python scripts/train.py data.batch_size=8         # 覆盖 batch size
    python scripts/train.py +experiment=my_exp        # 使用实验配置

配置路径:
    - 主配置: configs/train.yaml
    - 模型配置: configs/model/ (如 fastwam.yaml, fastwam_joint.yaml)
    - 数据配置: configs/data/ (如 bridge.yaml, lerobot.yaml)

该脚本在启动训练前会注册自定义的 OmegaConf 解析器（resolvers），
这些解析器支持配置中的计算表达式（如 `${add:${.dim},128}`）。
"""

import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    """
    Hydra 驱动的主函数，加载配置并启动训练。

    参数:
        cfg (DictConfig): Hydra/OmegaConf 提供的完整配置对象，
                          包含 model, data, training 等所有子配置。
    """
    run_training(cfg)


if __name__ == "__main__":
    main()
