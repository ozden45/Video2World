# --------------------------------------------------------------
#   mono_depth_midas.py
#
#   Description:
#       This script includes a module that extracts the depth 
#       from the frame using midas depth model.
#   
#   Author: �zden �zel
#   Created: 2026-01-29
#
# --------------------------------------------------------------


import cv2 as cv
import torch
from typing import Literal
import matplotlib.pyplot as plt


class MonocularDepthModel:
    def __init__(self, cam_params: dict, model_type: Literal["DPT_Large", "DPT_Hybrid", "MiDaS_small"] = "DPT_Large", device_type: Literal["cuda", "cpu"] = "cuda"):
        match device_type:
            case "cpu":
                self.device = torch.device("cpu")
                print("Cuda device not found, set as cpu")
            case "cuda":
                self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            case _:
                raise TypeError(f"Invalid device type {device_type}")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.cam_params = cam_params


    def __call__(self, img) -> torch.Tensor:
        input_batch = self.transform(img).to(self.device)
    
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
        min_depth, max_depth = self.cam_params["min_depth"], self.cam_params["max_depth"]

        return max_depth * pred + min_depth



if __name__ == "__main__":
    filename = r"C:\Users\ozden\source\repos\Video2World\src\models\dog.jpg"

    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cam_params = {"min_depth": 1, "min_depth": 1}

    model = MonocularDepthModel(cam_params, model_type="DPT_Large", device_type="cuda")
    depth_map = model(img)
    

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(depth_map.cpu().numpy(), cmap='plasma')






