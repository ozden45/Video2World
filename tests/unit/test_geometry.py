import torch
from v2w.geometry.projection import *
from v2w.geometry.points import *
from v2w.geometry.extrinsic_cam import *
from v2w.geometry.intrinsic_cam import *
from v2w.io import load_image, load_npy


def test_projection_output_shape():
    pts = torch.randn(5, 3)
    K = torch.eye(3)

    out = project_points(pts, K)

    assert out.shape == (5, 2)
    
    
    frame_path = Path("../data/dog.jpg")
    depth_path = Path("../data/depth.npy")
    
    frame = load_image(frame_path)
    depth = load_npy(depth_path)
    
    sfm_pts = ImagePoints(frame, depth)


#====================================
# Extrinsic cam test
#====================================

def test_extrinsic_cam():
    pass


#====================================
# Intrinsic cam test
#====================================

def test_extrinsic_cam():
    pass


