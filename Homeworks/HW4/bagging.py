__author__ = 'Allison MacLeay'

from sklearn.tree import DecisionTreeClassifier
import CS6140_A_MacLeay.Homeworks.HW4.data_load as dl
import numpy as np

class Bagging(object):
    def __init__(self, max_rounds=10, sample_size=10, learner=DecisionTreeClassifier):
        self.max_rounds = max_rounds
        self.sample_size = sample_size
        self.learner = learner
        self.predictions = []
        self.hypotheses = []
        self.train_error = 0

    def fit(self, X, y):
        for round in xrange(self.max_rounds):
            sub_X, sub_y = dl.random_sample(X, y, size=self.sample_size)
            hypothesis = self.learner().fit(sub_X, sub_y)
            pred_y = hypothesis.predict(sub_X)
            error = float(sum([0 if py == ty else 1 for py, ty in zip(pred_y, sub_y)]))/len(sub_y)
            print 'Round error: {}'.format(error)
            self.predictions.append(pred_y)
            self.hypotheses.append(hypothesis)
            pred_bagged = self.predict(sub_X)
            train_error = float(sum([0 if py == ty else 1 for py, ty in zip(pred_bagged, sub_y)]))/len(sub_y)
            print 'Bagged Train Error: {}'.format(train_error)

    def predict(self, X):
        h_pred = []
        for h in self.hypotheses:
            h_pred.append(h.predict_proba(X))
        return self.get_bagged_prediction(h_pred)

    def get_bagged_prediction(self, hpred):
        size = len(hpred[0])
        p_sum = np.zeros(size)
        for pred in hpred:
            pred = self.unzip_prob(pred)
            for p in range(len(pred)):
                p_sum[p] += pred[p]
        return [1 if float(p)/len(hpred) >= .5 else -1 for p in p_sum]

    def _check_y(self, y):
        if {1, 0}.issubset(set(y)):
            return y
        elif {-1, 1}.issubset(set(y)):
            return [1 if yi == 1 else 0 for yi in y]
        else:
            raise ValueError("Bad labels. Expected either 0/1 or -1/1, but got: {}".format(sorted(set(y))))


    def unzip_prob(self, y):
        # probability of being in class 1
        return list(zip(*y)[1])



