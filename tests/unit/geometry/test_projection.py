"""
tests.unit.geometry.test_projection

Tests for projection functions in v2w.geometry.projection module. 
"""

from v2w.geometry.projection import *
from v2w.geometry.points import *


def test_project_sfm_to_cam(sfm_pts, cam_pts, W):
    projected_cam_pts = project_sfm_to_cam(sfm_pts, W)
    
    assert cam_pts == projected_cam_pts


def test_project_cam_to_ray(cam_pts, ray_pts):
    projected_ray_pts = project_cam_to_ray(cam_pts)
    
    assert ray_pts == projected_ray_pts

"""
def test_project_ray_to_img(ray_pts, img_pts, K):
    projected_img_pts = project_ray_to_img(ray_pts, K)
    print(f"projected_img_pts: {projected_img_pts}")
    print(f"img_pts: {img_pts}")
    assert img_pts == projected_img_pts
""" 