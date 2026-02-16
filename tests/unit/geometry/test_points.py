"""
Docstring for tests.unit.geometry.test_points
"""

from v2w.geometry.points import *


class TestPoints:
    def test_add_point(self, p1, p2):
        pts = Points()
        pts += p1
        
        assert pts.coords.shape == (1, 3)
        assert pts.covariances.shape == (1, 3, 3)
        assert pts.colors.shape == (1, 3)
        assert pts.alphas.shape[0] == 1
        
        pts += p2
        
        assert pts.coords.shape == (2, 3)
        assert pts.covariances.shape == (2, 3, 3)
        assert pts.colors.shape == (2, 3)
        assert pts.alphas.shape[0] == 2
        
    def test_add_points(self, pts1, pts2):
        _pts = Points()
        _pts += pts1
        
        assert _pts.coords.shape == (2, 3)
        assert _pts.covariances.shape == (2, 3, 3)
        assert _pts.colors.shape == (2, 3)
        assert _pts.alphas.shape[0] == 2
        
        _pts += pts2
        
        assert _pts.coords.shape == (4, 3)
        assert _pts.covariances.shape == (4, 3, 3)
        assert _pts.colors.shape == (4, 3)
        assert _pts.alphas.shape[0] == 4
        

class TestPointCloud:
    def test_point_cloud_init(self, bounds, res):
        pts_cloud = PointCloud(bounds, res, "cuda")
        
        assert pts_cloud.get_filled_voxels() == None

    def test_point_cloud_bound(self, pts_cloud, bounds, res):
        dims = ((bounds[:, 1] - bounds[:, 0]) / res).floor().to(torch.long)
        print(f"dims: {dims}, pts_cloud.shape: {pts_cloud.shape}")
        
        assert pts_cloud.shape == dims




