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
        
    def test_eq_points(self, pts1, p1, p2):
        _pts = Points()
        _pts += p1
        _pts += p2
        print(pts1)
        print(_pts)
        assert pts1 == _pts
        

class TestPointCloud:
    def test_point_cloud_init(self, bounds, res):
        pts_cloud = PointCloud(bounds, res)
        
        assert pts_cloud.get_filled_voxels() == None

    def test_point_cloud_bound(self, pts_cloud, bounds, res):
        dims = ((bounds[:, 1] - bounds[:, 0]) / res).floor().to(torch.long)
        dims = dims.to(pts_cloud.device)
        
        assert torch.equal(pts_cloud.shape, dims)

    def test_add_points(self, pts_cloud, pts1, pts2):
        pts = Points()
        pts += pts1
        pts_cloud.add(pts1)
        
        assert pts == pts_cloud.get_filled_voxels()
        
        pts += pts2
        pts_cloud.add(pts2)
        
        
        print("==============================")
        print(pts)
        print("==============================")
        
        print("==============================")
        print(pts_cloud.get_filled_voxels())
        print("==============================")
        
        assert pts == pts_cloud.get_filled_voxels()
        
        

pts
Points(coords=tensor([[ 1.0000,  2.0000,  3.0000],
        [ 2.3000,  0.1000, -3.0000],
        [ 2.3000,  0.1000, -3.0000],
        [ 0.3000,  9.9000, -8.2000]], device='cuda:0'), covariances=tensor([[[0.5000, 0.3000, 0.4000],
         [0.1000, 0.1000, 0.2000],
         [0.5200, 0.1300, 0.4100]],

        [[0.5000, 0.3000, 0.4000],
         [0.1000, 0.1000, 0.2000],
         [0.5200, 0.1300, 0.4100]],

        [[0.5000, 0.3000, 0.4000],
         [0.1000, 0.1000, 0.2000],
         [0.5200, 0.1300, 0.4100]],

        [[0.5000, 0.3000, 0.4000],
         [0.1000, 0.1000, 0.2000],
         [0.5200, 0.1300, 0.4100]]], device='cuda:0'), colors=tensor([[121,  10, 204],
        [ 40,   1,  74],
        [ 40,   1,  74],
        [ 67,   1,   8]], device='cuda:0', dtype=torch.uint8), alphas=tensor([0.5000, 0.8000, 0.3000, 0.4000], device='cuda:0'))

