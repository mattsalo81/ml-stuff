import numpy as np

class IndexedDataset(object):
    """indexes data to allow fast scanning"""

    def __init__(self, x, y):
        """Takes design matrix for inputs and an array for labels
        input features must be continuous, labels must be -1 or 1"""
        self.x = x
        self.y = y
        self.input_dims = len(x.shape)
        self.index_features()
        self.dist = np.ones(len(y))
        self.dist /= len(y)



    def index_features(self):
        """This guy turned out to be easier than I'd thought
        - Thanks Numpy!"""
        self.indexed = np.argsort(self.x, axis=0)
    
    def get_sorted_indexes_for_feature(self, feature_no):
        return self.indexed[:,feature_no]
