import numpy as np

class IndexedDataset(object):
    """indexes data to allow fast scanning"""

    def __init__(self, x, y):
        """Takes design matrix for inputs and an array for labels
        input features must be continuous, labels must be -1 or 1"""
        self.x = x
        self.y = y
        self.input_dims = len(x.shape())
        self.


    def index_feature(self, feature_no)
        
