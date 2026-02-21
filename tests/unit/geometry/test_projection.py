"""
tests.unit.geometry.test_projection

Tests for projection functions in v2w.geometry.projection module. 
"""

from v2w.geometry.projection import *
from v2w.geometry.points import *


def test_project_sfm_to_cam(sfm_pts, cam_pts, W):
    projected_sfm_pts = project_sfm_to_cam(sfm_pts, W)
    
    assert cam_pts == projected_sfm_pts
    assert cam_pts.coords.shape == projected_sfm_pts.coords.shape
    assert cam_pts.covariances.shape == projected_sfm_pts.covariances.shape
    assert cam_pts.colors.shape == projected_sfm_pts.colors.shape
    assert cam_pts.alphas.shape == projected_sfm_pts.alphas.shape


def test_project_cam_to_ray(cam_pts, ray_pts):
    projected_ray_pts = project_cam_to_ray(cam_pts)
    
    assert ray_pts == projected_ray_pts
    assert ray_pts.coords.shape == projected_ray_pts.coords.shape
    assert ray_pts.covariances.shape == projected_ray_pts.covariances.shape
    assert ray_pts.colors.shape == projected_ray_pts.colors.shape
    assert ray_pts.alphas.shape == projected_ray_pts.alphas.shape

    
def test_project_ray_to_img(ray_pts, img_pts, K):
    projected_img_pts = project_ray_to_img(ray_pts, K)
    
    assert img_pts == projected_img_pts
    assert img_pts.coords.shape == projected_img_pts.coords.shape
    assert img_pts.covariances.shape == projected_img_pts.covariances.shape
    assert img_pts.colors.shape == projected_img_pts.colors.shape
    assert img_pts.alphas.shape == projected_img_pts.alphas.shape
    