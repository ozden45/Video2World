import torch
from v2w.geometry.points import *
from v2w.geometry.projection import *
from v2w.geometry.reconstruction import *
from v2w.io import load_intrinsic_mat
from v2w.config.loader import load_cam_config


def test_points_batched(cam_cfg_path):
    B = 4
    N = 1
    
    cfg = load_cam_config(cam_cfg_path)    
    K = load_intrinsic_mat(cfg)
    
    W_batched = torch.tensor([[[ 0.9966,  0.0751, -0.0331,  0.4392],
                               [ 0.0737, -0.9964, -0.0411, -0.2577],
                               [-0.0361,  0.0385, -0.9986,  1.2922],
                               [ 0.0000,  0.0000,  0.0000,  1.0000]],
                              [[ 0.9966,  0.0751, -0.0331,  0.4392],
                               [ 0.0737, -0.9964, -0.0411, -0.2577],
                               [-0.0361,  0.0385, -0.9986,  1.2922],
                               [ 0.0000,  0.0000,  0.0000,  1.0000]],
                              [[ 0.9966,  0.0751, -0.0331,  0.4392],
                               [ 0.0737, -0.9964, -0.0411, -0.2577],
                               [-0.0361,  0.0385, -0.9986,  1.2922],
                               [ 0.0000,  0.0000,  0.0000,  1.0000]],
                              [[ 0.9966,  0.0751, -0.0331,  0.4392],
                               [ 0.0737, -0.9964, -0.0411, -0.2577],
                               [-0.0361,  0.0385, -0.9986,  1.2922],
                               [ 0.0000,  0.0000,  0.0000,  1.0000]]])
    W_batched = W_batched[:, :3, :]
    
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
    
    
    cam_batched = project_sfm_to_cam_batched(sfm_batched, W_batched)
    
    img_batched = project_cam_to_img_batched(cam_batched, K)
    
    cam_batched_r = reconstruct_img_to_cam_batched(img_batched, K, cam_batched.coords[:, :, 2])
    
    assert cam_batched.extract_all_points() == cam_batched_r.extract_all_points()
    
    sfm_batched_r = reconstruct_cam_to_sfm_batched(cam_batched_r, W_batched)
    
    assert sfm_batched.extract_all_points() == sfm_batched_r.extract_all_points()
    