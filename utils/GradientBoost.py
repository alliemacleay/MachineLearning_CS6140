__author__ = 'Allison MacLeay'

from sklearn.tree import DecisionTreeRegressor
import CS6140_A_MacLeay.Homeworks.HW4 as hw4
import numpy as np

class GradientBoostRegressor(object):
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=1, learner=DecisionTreeRegressor):
        self.train_score = 0
        self.max_rounds = n_estimators
        self.learner = learner
        self.learning_rate = learning_rate #TODO - unused variable
        self.max_depth = max_depth
        self.hypotheses = []
        self.mean = None
        self.training_error = []
        self.local_error = []

    def fit(self, X, y):
        original_y = y[:]
        self.mean = np.mean(y)
        last_y = [self.mean for _ in range(len(y))]
        for round in xrange(self.max_rounds):
            residual = [-1. *(yn - yl) for yn, yl in zip(y, last_y)]
            hypothesis = self.learner().fit(X, residual)
            y = hypothesis.predict(X)
            last_y = [yn + self.mean for yn in y]
            self.local_error.append(hw4.compute_mse(last_y, original_y))
            self.hypotheses.append(hypothesis)
            pred_round = self.predict(X)
            self.train_score = hw4.compute_mse(pred_round, original_y)
            self.training_error.append(self.train_score)


    def predict(self, X):
        predictions = []
        p_sum = np.zeros(len(X))
        for h in self.hypotheses:
            predictions.append(h.predict(X))
        for pred_array in predictions:
            for p in range(len(pred_array)):
                p_sum[p] += pred_array[p]
        return [self.mean + p for p in p_sum]

    def print_stats(self):
        for r in range(len(self.training_error)):
            print 'Round {}: local error: {} training error: {}'.format(r, self.local_error[r], self.training_error[r])



    def decision_function(self):
        pass

    def loss(self, y, yhat, weights):
        return sum([(yh - yt)**2 for yh, yt, w in zip(yhat, y, weights)]) * .5

