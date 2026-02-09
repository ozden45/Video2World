import torch
from v2w.geometry.projection import *
from v2w.geometry.points import *
from v2w.geometry.extrinsic_cam import *
from v2w.geometry.intrinsic_cam import *


def test_projection_output_shape():
    pts = torch.randn(5, 3)
    K = torch.eye(3)

    out = project_points(pts, K)

    assert out.shape == (5, 2)
    
    
    
    sfm_pts = ImagePoints()
