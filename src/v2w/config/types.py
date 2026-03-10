"""
types.py
"""

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
class IntrinsicCameraConfig(BaseConfig):
    f_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    width_px: int
    height_px: int

@dataclass
class ExtrinsicCameraConfig(BaseConfig):
    csv_data_path: str
        
@dataclass
class CameraConfig(BaseConfig):
    intrinsic: IntrinsicCameraConfig
    extrinsic: ExtrinsicCameraConfig


#----------------------------------
# All configs
#----------------------------------

@dataclass
class Config(BaseConfig):
    #training: TrainConfig
    #model: ModelConfig
    #dataset: DatasetConfig
    camera: CameraConfig
    


