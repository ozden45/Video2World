from v2w.geometry.projection import *

def test_sfm_to_cam(sfm_pts, cam_pts, extrinsic):
    projected_sfm_pts = project_sfm_to_cam(cam_pts, extrinsic)
    print("===========================================")
    print(sfm_pts.coords)
    print(projected_sfm_pts.coords)
    print("===========================================")
    print(sfm_pts.covariances)
    print(projected_sfm_pts.covariances)
    assert sfm_pts == projected_sfm_pts
    
def test_cam_to_ray(cam_pts, ray_pts):
    pass

def test_ray_to_img(ray_pts, img_pts):
    pass

def test_sfm_to_img(sfm_pts, img_pts):
    pass