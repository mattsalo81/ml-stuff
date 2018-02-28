#!/home/syrup/anaconda3/envs/tensorflow/bin/python

import unittest
import numpy as np
from indexed_dataset import IndexedDataset

class BasicTests(unittest.TestCase):

    def test_indexing(self):
        x = np.array([[5, 10, 20],[3, 60, 9],[15, 13, 1]])
        y = np.array([-1, -1, 1])
        thing = IndexedDataset(x, y) 
        thing_i0 = thing.get_sorted_indexes_for_feature(0)
        thing_i1 = thing.get_sorted_indexes_for_feature(1)
        thing_i2 = thing.get_sorted_indexes_for_feature(2)
        self.assertListEqual(thing_i0.tolist(), [1, 0, 2])
        self.assertListEqual(thing_i1.tolist(), [0, 2, 1])
        self.assertListEqual(thing_i2.tolist(), [2, 1, 0])

if __name__ == '__main__':
    main()
