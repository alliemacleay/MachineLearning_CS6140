import numpy as np

__author__ = 'Allison MacLeay'

class DualPerceptron(object):
    def __init__(self, T=1):
        self.T = T
        self.alphas = None
        self.w = None
        self.y = None
        self.kernel = Kernel()


    def fit(self, X_train, y_train, w=None, alphas=None):
        if type(X_train) is not np.ndarray:
            X_train = np.array(X_train)
        if type(y_train) is not np.ndarray:
            y_train = np.array(y_train)
        self.y_train = y_train
        self.X_train = X_train

    def decision_function(self, X, w=None, alphas=None):
        y_pred = np.zeros(X.shape[0])
        K = self.calc_k(X)
        n_rows, n_cols = X.shape

        self.alphas = np.zeros(self.X_train.shape[0]) if alphas is None else alphas

        K = self.calc_k(X)

        for t in xrange(self.T):
            mistakes = 0
            for j in xrange(self.X_train.shape[0]):
                sum_x = 0
                for i in xrange(X.shape[0]):
                    sum_x += self.y_train[j] * self.alphas[j] * K[i, j]
                if sum_x <= 0:
                    self.alphas[j] += 1
                    mistakes += 1
            print 'iteration {} mistakes {}'.format(self.T, mistakes)
            if mistakes == 0:
                t = self.T

        for i in xrange(X.shape[0]):
            sum_x = 0
            for j in xrange(self.X_train.shape[0]):
                sum_x += self.y_train[i] * self.alphas[i] * K[i, j]
            y_pred[i] = sum_x

        return y_pred

    def predict(self, X):
        return np.sign(self.decision_function(X))


    def calc_k(self, X_test):
        n_rows = X_test.shape[0]
        n_cols = self.X_train.shape[0]
        K = np.zeros((n_rows, n_cols))
        for i in xrange(n_rows):
            for j in xrange(n_rows):
                K[i, j] = self.kernel.f(X_test, self.X_train, i, j)
        return K

class Kernel(object):
    def __init__(self, ktype='dot'):
        self.type = ktype

    def f(self, x, y, i, j):
        if self.type == 'dot':
            return np.dot(x[i], y[j])
        else:
            return np.sqrt(np.sum((x[i] - y[j]) ** 2))

