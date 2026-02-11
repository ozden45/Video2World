# --------------------------------------------------------------
#   config_classes.py
#
#   Description:
#
#   Author: Özden Özel
#   Created: 2026-02-08
#
# --------------------------------------------------------------

from dataclasses import dataclass
from v2w.config.base import BaseConfig

#----------------------------------
# Model configs
#----------------------------------

@dataclass
class ModelConfig(BaseConfig):
    pass


#----------------------------------
# Training configs
#----------------------------------

@dataclass
class TrainConfig(BaseConfig):
    batch_size: int
    epochs: int

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")


#----------------------------------
# Dataset configs
#----------------------------------

@dataclass
class DatasetConfig(BaseConfig):
    pass


#----------------------------------
# Camera configs
#----------------------------------

@dataclass
class IntrinsicCamConfig(BaseConfig):
    f_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    width_px: int
    height_px: int

@dataclass
class ExtrinsicCamConfig(BaseConfig):
    csv_data_path: str
        
@dataclass
class CamConfig(BaseConfig):
    intrinsic: IntrinsicCamConfig
    extrinsic: ExtrinsicCamConfig


#----------------------------------
# All configs
#----------------------------------

@dataclass
class Config(BaseConfig):
    training: TrainConfig
    model: ModelConfig
    dataset: DatasetConfig
    camera: CamConfig
    


