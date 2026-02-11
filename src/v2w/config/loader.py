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
from v2w.config.types import Config
from v2w.io import load_yaml
from v2w.utils.misc import _deep_merge, if_path_exists


def load_config(path: str | Path) -> Config:
    """Load single yaml file."""
    
    # Check that path exists
    if not if_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
        
    raw = load_yaml(path)
    return Config.from_dict(raw)


def load_many_config(paths: list[str | Path]) -> Config:
    """Load and merge multiple yaml files."""
    merged = {}

    for path in paths:
        # Check that path exists
        if not if_path_exists(path):
            raise FileNotFoundError(f"The path {path} is not found")
        
        merged = _deep_merge(merged, _read_yaml(p))

    return Config.from_dict(merged)


def save_config(cfg: Config, path: str | Path):
    """Save config to yaml (for experiment reproducibility)."""
    
    # Check that path exists
    if not if_path_exists(path):
        raise FileNotFoundError(f"The path {path} is not found")
    
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f)

