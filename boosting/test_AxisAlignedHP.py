#!/home/syrup/anaconda3/envs/tensorflow/bin/python

import unittest
import numpy as np
from axis_aligned_hyperplane_classifier import AxisAlignedHyperplane

class BasicAAHPTests(unittest.TestCase):

    def test_init(self):
        hp = AxisAlignedHyperplane(0, 0)

    def test_scalar_classification(self):
        hp = AxisAlignedHyperplane(0, 0)
        self.assertEqual(1, hp.classify_value(10))
        self.assertEqual(-1, hp.classify_value(-10))
        self.assertEqual(1, hp.classify_value(0))

        hp = AxisAlignedHyperplane(0, 0, direction=-1)
        self.assertEqual(1, hp.classify_value(-5))
        self.assertEqual(-1, hp.classify_value(10))
        self.assertEqual(-1, hp.classify_value(0))

        hp = AxisAlignedHyperplane(0, 10, direction=1)
        self.assertEqual(-1, hp.classify_value(5))
        self.assertEqual(1, hp.classify_value(100))
        self.assertEqual(1, hp.classify_value(10))

    def test_vector_classification(self):
        hp = AxisAlignedHyperplane(0, 0)
        vector = [1, 2, 3]
        self.assertEqual(1, hp.classify_vector(vector))
        vector = [1, -2, -3]
        self.assertEqual(1, hp.classify_vector(vector))
        vector = [-1, 2, 3]
        self.assertEqual(-1, hp.classify_vector(vector))

        hp = AxisAlignedHyperplane(1, 0)
        vector = [1, 2, 3]
        self.assertEqual(1, hp.classify_vector(vector))
        vector = [1, -2, -3]
        self.assertEqual(-1, hp.classify_vector(vector))
        vector = [-1, 2, 3]
        self.assertEqual(1, hp.classify_vector(vector))

    def test_matrix_classification(self):
        hp = AxisAlignedHyperplane(0, 0)
        matrix = np.array([[-1, 1, 2], [100, 200, 1], [1, 2, 3]])
        self.assertListEqual([-1, 1, 1], hp.classify_matrix(matrix).tolist())


if __name__ == '__main__':
    main()
