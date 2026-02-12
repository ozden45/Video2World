"""
Docstring for tests.unit.geometry.test_points
"""

from v2w.geometry.points import *


class TestPoints:
    def test_add_point(self, p1, p2):
        pts = Points()
        pts += p1
        print("==============================================")
        print(f"p1.coords.shape: {p1.coords.shape}")
        print(f"p1.covariance.shape: {p1.covariance.shape}")
        print(f"p1.color.shape: {p1.color.shape}")
        print(f"p1.alpha.shape: {p1.alpha.shape}")
        print("==============================================")
        print(f"pts.coords.shape: {pts.coords.shape}")
        print(f"pts.covariance.shape: {pts.covariances.shape}")
        print(f"pts.color.shape: {pts.colors.shape}")
        print(f"pts.alpha.shape: {pts.alphas.shape}")
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
    pass





