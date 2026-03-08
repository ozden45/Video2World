import torch
import logging
from typing import Literal
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class MonocularDepthModel:
    model_type: Literal["DPT_Large", "DPT_Hybrid", "MiDaS_small"] = "MiDaS_small"
    input_dim: torch.Tensor = (512, 512)
    device: torch.device = None
    min_depth: float = 0
    max_depth: float = 1
    
    def __post_init__(self):
        """
        """
        
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.to(self.device)

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform


    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        input_batch = self.transform(img).to(self.device)
    
        try:
            with torch.no_grad():
                pred = self.model(input_batch)

                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Normalize the depth
            pred = (pred - pred.amin()) / (pred.amax() - pred.amin())
        except Exception as e:
            logger.warning(f"Depth estimation failed: {e}")
        
        return self.max_depth * pred + self.min_depth
    

    def _resolve_device(self, device):
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
        
    def _resolve_dtype(self, dtype):
        if dtype is None:
            return torch.float32
        else:
            return torch.as_tensor(1, dtype=dtype).dtype
        
