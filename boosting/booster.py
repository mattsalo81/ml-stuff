import numpy as np
import indexed_dataset as i_d

class Booster(object):
    """creates a boosting model.  Keeps a list of axis-aligned-hyperplanes,
    along with their general weights."""

    def __init__(self, x, y):
        self.data = i_d.IndexedDataset(x, y)
        self.x = x
        self.y = y
        self.learners = []
        self.alpha = []

    def train(self, epochs):


        for epoch in range(epochs):
            
    def add_weak_learner(self):
        learner = self.data.best_bisection(self.dist)
        

    def update_weights(self, learner):


