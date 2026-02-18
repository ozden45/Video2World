"""
tests.unit.geometry.test_projection

Tests for projection functions in v2w.geometry.projection module. 
"""

from v2w.geometry.projection import *
from v2w.geometry.points import *


def test_project_sfm_to_cam(sfm_pts, cam_pts, W):
    cam_pts = project_sfm_to_cam(sfm_pts, W)
    
    assert cam_pts.coords.shape == sfm_pts.coords.shape
    assert cam_pts.covariances.shape == sfm_pts.covariances.shape
    assert cam_pts.colors.shape == sfm_pts.colors.shape
    assert cam_pts.alphas.shape == sfm_pts.alphas.shape
        
def test_project_cam_to_ray(cam_pts, ray_pts):
    ray_pts = project_cam_to_ray(cam_pts)
    
    assert ray_pts.coords.shape == cam_pts.coords.shape
    assert ray_pts.covariances.shape == cam_pts.covariances.shape
    assert ray_pts.colors.shape == cam_pts.colors.shape
    assert ray_pts.alphas.shape == cam_pts.alphas.shape

def test_project_ray_to_img(ray_pts, img_pts):
    img_pts = project_ray_to_img(ray_pts)
    
    assert img_pts.coords.shape == ray_pts.coords.shape
    assert img_pts.covariances.shape == ray_pts.covariances.shape
    assert img_pts.colors.shape == ray_pts.colors.shape
    assert img_pts.alphas.shape == ray_pts.alphas.shape
    