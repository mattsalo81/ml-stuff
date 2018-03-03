import numpy as np
import axis_aligned_hyperplane_classifier as ahp

class IndexedDataset(object):
    """indexes data to allow fast scanning"""
    # pandas has something called a dataframe that I should look into

    def __init__(self, x, y):
        """Takes design matrix for inputs and an array for labels
        input features must be continuous, labels must be -1 or 1
        and will be reduced if not -1 or 1"""
        self.x = x
        self.orig_y = y
        self.y = np.copy(self.orig_y)
        self.input_dims = len(x.shape)
        self.index_features()
        # normalize labels
        y[y>=0] = 1
        y[y<0] = -1
        # current weights
        self.dist = np.ones(y.shape) / np.size(y, axis=0)
        self.learners = []
        self.alpha = []

    def index_features(self):
        """This guy turned out to be easier than I'd thought
        - Thanks Numpy!"""
        self.indexed = np.argsort(self.x, axis=0)
    
    def get_sorted_indexes_for_feature(self, feature_no):
        return self.indexed[:,feature_no]
    
    def classify_value(self, value):
        score = 0
        for i in range(len(self.learners)):
            score_i = self.learners[i].classify_value(value)
            score_i *= self.alpha[i]
            score += score_i
        return np.sign(score)

    def classify_vector(self, vector):
        score = 0
        for i in range(len(self.learners)):
            score_i = self.learners[i].classify_vector(vector)
            score_i *= self.alpha[i]
            score += score_i
        return np.sign(score)

    def classify_matrix(self, matrix):
        score = np.zeros(np.size(matrix, axis=0))
        for i in range(len(self.learners)):
            score_i = self.learners[i].classify_matrix(matrix)
            score_i = np.multiply(score_i, self.alpha[i])
            score += score_i
        return np.sign(score)

    def new_learner(self):
        learner = self.best_bisection()
        pred = learner.classify_matrix(self.x)
        correct = np.multiply(pred, self.y)
        error = np.minimum(correct, 0) 
        error = np.multiply(error, self.dist)
        error = -np.sum(error)
        temp = (1 - error) / error

        alpha = (1 / 2) * np.log(temp)

        # update distribution
        self.dist *= np.exp(-alpha * correct)
        self.dist /= np.sum(self.dist)
        self.learners.append(learner)
        self.alpha.append(alpha)

    def best_bisection(self):
        """searches all values in all features for the best bisection along
        an axis aligned hyperplane.""" 
        # set max vars
        max_acc = -float("inf")
        max_acc_bis = -float("inf")
        max_acc_feat = 0
        max_acc_dir = 1

        for feature_no in range(np.size(self.x, axis=1)):
            # iterate every feature
            sorted_index = self.get_sorted_indexes_for_feature(feature_no)
            sorted_vals = self.x[:, feature_no][sorted_index]
            ave_vals = [-float("inf")]
            ave_vals.extend(list(sorted_vals[1:] + sorted_vals[:-1]))
            ave_vals.append(float("inf"))
            ave_vals = [0.5 * float(i) for i in ave_vals]
            for bisect_value in ave_vals:
                # iterate over every value, sorted low to high.
                plane = ahp.AxisAlignedHyperplane(feature_no, 
                                                  bisect_value,
                                                  direction=1)
                pred = plane.classify_matrix(self.x)
                correct = np.multiply(pred, self.y)
                weighted_results = np.multiply(correct, self.dist)
                accuracy = np.sum(weighted_results)

                # check if best accuracy
                if accuracy > max_acc:
                    max_acc = accuracy
                    max_acc_bis = bisect_value
                    max_acc_feat = feature_no
                    max_acc_dir = 1
                accuracy *= -1
                # check if best accuracy if everything flipped
                if accuracy > max_acc:
                    max_acc = accuracy
                    max_acc_bis = bisect_value
                    max_acc_feat = feature_no
                    max_acc_dir = -1


        return ahp.AxisAlignedHyperplane(max_acc_feat, 
                                         max_acc_bis, 
                                         direction=max_acc_dir)

        

            

