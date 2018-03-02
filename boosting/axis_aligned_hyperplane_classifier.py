#!/home/syrup/anaconda3/envs/tensorflow/bin/python
import numpy as np

class AxisAlignedHyperplane(object):
    """An AxisAlignedHyperPlane bisects feature space into a positive side
    and a negative side.  It intersects exactly one axis at one value.  The
    positive side (higher vals in feature space) is where positive 
    values/labels are assumed to be.  if direction is set to -1, then this is
    reversed, with positive values/labels assumed to be in the lower section.
    
    any points lying on the hyperplane are assumed to be positive if direction
    is positive, and negative if direction is negative.  This means that 
    switching the direction of a hyperplane reverses the classification of
    every single point in hyperspace."""

    def __init__(self, feature_no, bisect_value, direction=1):
        self.feature_no = feature_no
        self.bisect_value = bisect_value
        self.direction = np.sign(direction)

    def classify_value(self, value):
        """classifies value given as if it were the coordinate in feature 
        space on the axis we're bisecting.  returns +1 or -1"""
        if value >= self.bisect_value:
            classification = 1
        else:
            classification = -1
        return classification * self.direction

    def classify_vector(self, vector):
        """Takes a vector of features and classifies it based on the stored
        feature_no.  returns +1 or -1"""
        return self.classify_value(vector[self.feature_no])

    def classify_matrix(self, matrix):
        """Takes a design matrix and returns a classified vector based on
        the stored feature no"""
        return np.apply_along_axis(self.classify_vector, 1, matrix)
