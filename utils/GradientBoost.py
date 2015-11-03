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
        X = np.asarray(X)
        y = np.asarray(y)

        self.mean = np.mean(y)
        for round in xrange(self.max_rounds):
            residual = [(yn - yl) for yn, yl in zip(y, self.predict(X))]
            hypothesis = self.learner().fit(X, residual)
            self.hypotheses.append(hypothesis)

            self.local_error.append(hw4.compute_mse(residual, hypothesis.predict(X)))

            pred_round = self.predict(X)
            self.train_score = hw4.compute_mse(pred_round, y)
            self.training_error.append(self.train_score)

    def predict(self, X):
        X = np.asarray(X)
        predictions = np.array([self.mean] * X.shape[0])
        for h in self.hypotheses:
            predictions += h.predict(X)
        return predictions

    def print_stats(self):
        for r in range(len(self.training_error)):
            print 'Round {}: local error: {} training error: {}'.format(r, self.local_error[r], self.training_error[r])



    def decision_function(self):
        pass

    def loss(self, y, yhat, weights):
        return sum([(yh - yt)**2 for yh, yt, w in zip(yhat, y, weights)]) * .5

