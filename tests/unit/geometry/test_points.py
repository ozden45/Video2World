"""
Docstring for tests.unit.geometry.test_points
"""

from v2w.geometry.points import *


class TestPoint:
    def test_eq_point_1(self, p1, p2):
        assert p1 != p2
        
    def test_eq_point_2(self, p1):
        p2 = Point(
            coords = torch.tensor([1, 2, 3]),
            covariance = torch.tensor([[0.5, 0.3, 0.4], [0.1, 0.1, 0.2], [0.52, 0.13, 0.41]]),
            color = torch.tensor([121, 10, 204]),
            alpha = torch.tensor([0.5])
        )
        
        assert p1 == p2
        

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
        
    def test_eq_points(self, p1, p2):
        _pts1 = Points()
        _pts1 += p1
        _pts1 += p2
        
        _pts2 = Points()
        _pts2 += p1
        _pts2 += p2
        
        assert _pts1 == _pts2
        

class TestPointCloud:
    def test_point_cloud_init(self, bounds, res):
        pts_cloud = PointCloud(bounds, res)
        
        assert pts_cloud.get_filled_voxels() == None

    def test_point_cloud_bound(self, pts_cloud, bounds, res):
        dims = ((bounds[:, 1] - bounds[:, 0]) / res).floor().to(torch.long)
        dims = dims.to(pts_cloud.device)
        
        assert torch.equal(pts_cloud.shape, dims)

    # FIXME: This test ignores point duplications. This 
    # is a problem for the current implementation of 
    # PointCloud.add() method, which uses torch.unique() 
    # to remove duplicates. This can lead to unexpected 
    # behavior when adding the same points multiple times.
    """
    def test_add_points(self, pts_cloud, pts1, pts2):
        pts = Points()
        pts += pts1
        pts_cloud.add(pts1)
        
        assert pts == pts_cloud.get_filled_voxels()
        
        pts += pts2
        pts_cloud.add(pts2)
        
        assert pts == pts_cloud.get_filled_voxels()
    """ 
    def test_add_points_override(self, pts_cloud, pts1):
        pts = Points()
        pts += pts1
        pts_cloud.add(pts1)
        
        assert pts == pts_cloud.get_filled_voxels()
        
        pts += pts1
        pts_cloud.add(pts1)
        
        assert pts != pts_cloud.get_filled_voxels()