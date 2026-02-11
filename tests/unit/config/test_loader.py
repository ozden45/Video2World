"""
Docstring for tests.unit.config.test_loader
"""


from v2w.config.loader import *


class TestLoader:
    def test_loader(self):
        config_paths = [
            "configs/cam.yaml",
            #"configs/dataset.yaml",
            #"configs/default.yaml",
            #"configs/model.yaml",
            #"configs/train.yaml"
        ]
        
        for path in config_paths:
            cfg = load_config(path)
            cfg.
    
    def test_load_config(self):
        assert True
        
    def test_load_many_config(self):
        assert True
        
    def test_save_config(self):
        assert True
    
    
    
    