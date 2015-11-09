from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import CS6140_A_MacLeay.Homeworks.HW4 as hw4
import numpy as np


class AdaboostOptimal(object):
    def __init__(self, max_rounds=10, learner=DecisionTreeClassifier, verbose=False):
        self.max_rounds = max_rounds
        self.learner = learner
        self.verbose = verbose

        # Properties estimated from data
        self.hypotheses = []
        self.stump = []
        self.alpha = []
        self.snapshots = []
        self.adaboost_error = {}
        self.adaboost_error_test = {}
        self.local_errors = {}

        self.margin = 0
        self.feature_margin = 0
        self.abs_sum_alpha = 0

    def print_stats(self):
        pass

    def clone(self):
        ab = AdaboostOptimal(max_rounds=self.max_rounds, learner=self.learner, verbose=self.verbose)
        ab.hypotheses = list(self.hypotheses)
        ab.alpha = list(self.alpha)
        return ab

    def _check_y(self, y):
        if {1, 0}.issubset(set(y)):
            return y
        elif {-1, 1}.issubset(set(y)):
            return [1 if yi == 1 else 0 for yi in y]
        else:
            raise ValueError("Bad labels. Expected either 0/1 or -1/1, but got: {}".format(sorted(set(y))))

    def _check_y_not_zero(self, y):
        if {1, 0}.issubset(set(y)):
            return [1 if yi == 1 else -1 for yi in y]
        elif {-1, 1}.issubset(set(y)):
            return y
        else:
            raise ValueError("Bad labels. Expected either 0/1 or -1/1, but got: {}".format(sorted(set(y))))


    def fit(self, X, y):
        y = self._check_y(np.asarray(y))
        X = np.asarray(X)

        w = np.ones(X.shape[0]) / X.shape[0]
        for round_number in xrange(self.max_rounds):
            #print w
            current_hypothesis = self.learner().fit(X, y, sample_weight=w)
            y_pred = current_hypothesis.predict(X)
            #if round_number % 10 == 0:
            #    print round_number

            y = np.array(y)
            self.local_errors[round_number + 1] = float(np.sum([y==y_pred]))/y.shape[0]

            error = np.sum(w[y!=y_pred])
            #error = np.sum(w * (y!=y_pred))


            w[y == y_pred] *= error / (1.0 - error)
            w /= np.sum(w)

            #self.stump.append(hw4.DecisionStump(current_hypothesis.tree_.feature[0], current_hypothesis.tree_.threshold[0]))
            self.hypotheses.append(current_hypothesis)
            self.alpha.append(np.log((1 - error) / error) if error > 0 else 1.0)
            #self.snapshots.append(self.clone())

            #train_y = self.predict(X)
            #y = self._check_y(y)
            #train_y = self._check_y(train_y)
            #self.adaboost_error[round_number + 1] = float(np.sum([y == train_y]))/y.shape[0]

            #m = sum([wi * py for wi, py in zip(w, self._check_y_not_zero(train_y))])

            #self.abs_sum_alpha += np.abs(sum(w))
            #self.feature_margin = float(m)/self.abs_sum_alpha
            #self.margin += m

            if self.verbose:
                print("round {}: error={:.2f}. alpha={:.2f}. AUC={:.3f}".format(
                    round_number + 1, error, self.alpha[-1], roc_auc_score(y, self.predict(X))))

            if error == 0:   # converged early
                break

        return self

    def rank(self, X, theta=.5):
        delta = zip(np.abs(self.decision_function(X) - .5), range(len(X)))
        delta.sort()
        return zip(*delta)[1]


    def predict(self, X, theta=0.5):
        return [1 if score > theta else 0 for score in self.decision_function(X)]  # weighted majority

    def decision_function(self, X):
        X = np.asarray(X)
        scores = np.zeros(X.shape[0])
        for alpha, hypothesis in zip(self.alpha, self.hypotheses):
            scores += alpha * hypothesis.predict(X)
        return scores / np.sum(self.alpha)


