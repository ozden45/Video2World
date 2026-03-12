import pytest
import torch
from v2w.geometry.points import *
from v2w.geometry.projection import *
from v2w.geometry.reconstruction import *
from v2w.dataloaders import create_tumvi_dataloader


def test_points_batched():
    B = 4
    N = 100
    
    
    
    loader = create_tumvi_dataloader(
        root="/home/ozden/repos/Video2World/data/tum_vi",
        sequence="dataset-corridor4_512_16",
        batch_size=B
    )
    
    for batch in loader:
        print(batch)
        W_batched = batch["extrinsic"]
        break
    
    sfm_coords = torch.rand((B, N, 3))
    sfm_covs = torch.rand((B, N, 3, 3))
    sfm_colors = torch.rand(B, N, 3)
    sfm_alphas = torch.rand(B, N)
    
    sfm_batched = SFMPointsBatched(
        coords=sfm_coords,
        covariances=sfm_covs,
        colors=sfm_colors,
        alphas=sfm_alphas
    )
    
    
    cam_batched = project_sfm_to_cam_batched(sfm_batched, )

