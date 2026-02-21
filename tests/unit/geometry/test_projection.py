"""
tests.unit.geometry.test_projection

Tests for projection functions in v2w.geometry.projection module. 
"""

from v2w.geometry.projection import *
from v2w.geometry.points import *


def test_project_sfm_to_cam(sfm_pts, cam_pts1, cam_pts2, W1, W2):
    projected_sfm_pts = project_sfm_to_cam(sfm_pts, W1)
    print(f"cam_pts1: {cam_pts1}\n")
    print(f"projected_sfm_pts: {projected_sfm_pts}")
    assert cam_pts1 == projected_sfm_pts
    assert cam_pts1.coords.shape == projected_sfm_pts.coords.shape
    assert cam_pts1.covariances.shape == projected_sfm_pts.covariances.shape
    assert cam_pts1.colors.shape == projected_sfm_pts.colors.shape
    assert cam_pts1.alphas.shape == projected_sfm_pts.alphas.shape
    
    projected_sfm_pts = project_sfm_to_cam(sfm_pts, W2)
    assert cam_pts2 == projected_sfm_pts
    assert cam_pts2.coords.shape == projected_sfm_pts.coords.shape
    assert cam_pts2.covariances.shape == projected_sfm_pts.covariances.shape
    assert cam_pts2.colors.shape == projected_sfm_pts.colors.shape
    assert cam_pts2.alphas.shape == projected_sfm_pts.alphas.shape
    

def test_project_cam_to_ray(cam_pts1, cam_pts2, ray_pts1, ray_pts2):
    projected_ray_pts = project_cam_to_ray(cam_pts1)
    
    assert ray_pts1.coords.shape == projected_ray_pts.coords.shape
    assert ray_pts1.covariances.shape == projected_ray_pts.covariances.shape
    assert ray_pts1.colors.shape == projected_ray_pts.colors.shape
    assert ray_pts1.alphas.shape == projected_ray_pts.alphas.shape
    
    projected_ray_pts = project_cam_to_ray(cam_pts2)
    
    assert ray_pts2.coords.shape == projected_ray_pts.coords.shape
    assert ray_pts2.covariances.shape == projected_ray_pts.covariances.shape
    assert ray_pts2.colors.shape == projected_ray_pts.colors.shape
    assert ray_pts2.alphas.shape == projected_ray_pts.alphas.shape
    
    
"""
def test_project_ray_to_img(ray_pts1, ray_pts2, img_pts1, img_pts2, K):
    projected_img_pts = project_ray_to_img(ray_pts1, K)
    
    assert img_pts1.coords.shape == projected_img_pts.coords.shape
    assert img_pts1.covariances.shape == projected_img_pts.covariances.shape
    assert img_pts1.colors.shape == projected_img_pts.colors.shape
    assert img_pts1.alphas.shape == projected_img_pts.alphas.shape
    
    projected_img_pts = project_ray_to_img(ray_pts2, K)
    
    assert img_pts2.coords.shape == projected_img_pts.coords.shape
    assert img_pts2.covariances.shape == projected_img_pts.covariances.shape
    assert img_pts2.colors.shape == projected_img_pts.colors.shape
    assert img_pts2.alphas.shape == projected_img_pts.alphas.shape
"""