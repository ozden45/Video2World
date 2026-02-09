"""
Video2World (v2w)

Core API for training, inference and rendering.
"""

__version__ = "0.1.0"

__all__ = [
    # config
    "load_config",

    # datasets
    "Dataset",

    # models
    "Video2World",
    "GaussianSplatting",
    "MiDaSDepth",

    # rendering
    "Rasterizer",
]


def __getattr__(name):
    # -------- config --------
    if name == "load_config":
        from .config.loader import load_config
        return load_config

    # -------- datasets --------
    if name == "Dataset":
        from .datasets import Dataset
        return Dataset

    # -------- models --------
    if name == "Video2World":
        from .models.video2world import Video2World
        return Video2World

    if name == "GaussianSplatting":
        from .models.gaussian_splatting import GaussianSplatting
        return GaussianSplatting

    if name == "MiDaSDepth":
        from .models.mono_depth_midas import MiDaSDepth
        return MiDaSDepth

    # -------- rendering --------
    if name == "Rasterizer":
        from .rendering.rasterizer import Rasterizer
        return Rasterizer

    raise AttributeError(name)
