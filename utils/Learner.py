from Tests.hw3_tests import get_test_data_bayes

__author__ = 'Allison MacLeay'

import numpy as np
import CS6140_A_MacLeay.utils as utils


class Estimator(object):
    def fit(self, X, y):
        raise NotImplementedError

    def decision_function(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError


class NBBase(Estimator):
    def __init__(self):
        self.class_priors = None

    def fit(self, X, y):
        prior = np.sum(y) / len(y)
        self.class_priors = [1.0 - prior, prior]

    def get_log_conditional_prob(self, x, j, y):
        raise NotImplementedError

    def decision_function(self, X):
        return np.asarray([self.decision_function_one(x) for x in X])

    def predict(self, X):
        return np.asarray([1 if prob[1] > prob[0] else 0
                           for prob in self.decision_function(X)])

    def decision_function_one(self, x):
        log_p_star_one = np.sum([self.get_log_conditional_prob(x, j, 1) for j, xj in enumerate(x)])
        log_p_star_one += np.log(self.class_priors[1])
        p_star_one = np.exp(log_p_star_one)

        log_p_star_zero = np.sum([self.get_log_conditional_prob(x, j, 0) for j, xj in enumerate(x)])
        log_p_star_zero += np.log(self.class_priors[0])
        p_star_zero = np.exp(log_p_star_zero)

        Z = p_star_one + p_star_zero
        return p_star_zero / Z

    def transform(self):
        raise NotImplementedError

class BernoulliNB(NBBase):
    def fit(self, X, y):
        def get_probs(X):
            num_over = np.sum([1.0 if val > self.means[j] else 0.0 for val in X[:, j]])
            num_below = X.shape[0] - num_over
            prob_over = num_over / (num_over + num_below)
            prob_below = num_below / (num_over + num_below)
            return prob_over

        super(BernoulliNB, self).fit(X, y)
        self.means = np.mean(X, axis=0)
        self.probs_over_one = np.zeros(shape=X.shape[1])
        self.probs_over_zero = np.zeros(shape=X.shape[1])
        X_pos = X[y == 1]
        X_neg = X[y == 0]

        for j in range(X.shape[1]):
            self.probs_over_one[j] = get_probs(X_pos)
            self.probs_over_zero[j] = get_probs(X_neg)

    def get_log_conditional_prob(self, x, j, y):
        """"""
        def helper():
            if y == 0:
                if x[j] > self.means[j]:
                    return self.probs_over_zero[j]
                else:
                    return 1.0 - self.probs_over_zero[j]
            elif y == 1:
                if x[j] > self.means[j]:
                    return self.probs_over_one[j]
                else:
                    return 1.0 - self.probs_over_one[j]
        val = helper()
        return np.log(val if val != 0 else 0.01)


def main():
    X, y = get_test_data_bayes()
    nb = BernoulliNB()
    nb.fit(X, y)
    print(nb.decision_function(X))

if __name__ == "__main__":
    main()


