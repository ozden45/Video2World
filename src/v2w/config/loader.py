"""
Configuration loading utilities.

Responsible for:
- reading yaml
- merging multiple configs
- converting to dataclasses
- saving final config
"""


from pathlib import Path
import yaml
from dataclasses import asdict
from v2w.config.types import CamConfig, DatasetConfig, ModelConfig, TrainConfig
from v2w.io import load_yaml
from v2w.utils.misc import is_path_exists



def load_cam_config(path: str | Path) -> CamConfig:
    """Load camera configuration"""
    
    # Check that path exists
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")

    raw = load_yaml(path)
    return CamConfig.from_dict(raw)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load camera configuration"""
    
    # Check that path exists
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")

    raw = load_yaml(path)
    return DatasetConfig.from_dict(raw)


def load_model_config(path: str | Path) -> ModelConfig:
    """Load camera configuration"""
    
    # Check that path exists
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")

    raw = load_yaml(path)
    return ModelConfig.from_dict(raw)


def load_train_config(path: str | Path) -> TrainConfig:
    """Load camera configuration"""
    
    # Check that path exists
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")

    raw = load_yaml(path)
    return TrainConfig.from_dict(raw)


def load_config(path: str | Path) -> Config:
    """Load single yaml file."""
    
    # Check that path exists
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
        
    raw = load_yaml(path)
    return Config.from_dict(raw)


def save_config(cfg: Config, path: str | Path):
    """Save config to yaml for experiment reproducibility."""
    
    # Check that path exists
    if not is_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f)

