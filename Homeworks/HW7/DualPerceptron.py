import numpy as np

__author__ = 'Allison MacLeay'

class DualPerceptron(object):
    def __init__(self, T=1, kname='dot'):
        self.T = T
        self.alphas = None
        self.w = None
        self.y = None
        self.kernel = Kernel(kname)


    def _add_bias(self, X):
        bias = np.ones(X.shape[0])
        bias = bias.reshape((len(bias), 1))
        return np.concatenate([X, bias], axis=1)


    def fit(self, X_train, y_train, w=None, alphas=None):
        if type(X_train) is not np.ndarray:
            X_train = np.array(X_train)
        if type(y_train) is not np.ndarray:
            y_train = np.array(y_train)

        X_train = self._add_bias(X_train)

        self.y_train = y_train
        self.X_train = X_train
        print(self.kernel.type)
        K = self.calc_k(X_train, X_train)
        self.alphas = np.zeros(self.X_train.shape[0]) if alphas is None else alphas
        t = 0
        mistakes = 1
        while t < self.T and mistakes > 0:
            mistakes = 0
            for j in xrange(K.shape[1]):
                sum_x = 0
                for i in xrange(K.shape[0]):
                    sum_x += self.y_train[i] * self.alphas[i] * K[i, j]
                sum_x *= self.y_train[j]

                if sum_x <= 0:
                    self.alphas[j] += 1
                    mistakes += 1
            t += 1
            perc = float(mistakes)/X_train.shape[0] * 100
            print 'iteration {} mistakes {} ({}%)'.format(t, mistakes, perc)

    def decision_function(self, X, w=None, alphas=None):
        X = self._add_bias(X)

        y_pred = np.zeros(X.shape[0])
        K = self.calc_k(self.X_train, X)
        n_rows, n_cols = K.shape

        for i in xrange(n_cols):
            sum_x = 0
            for j in xrange(n_rows):
                sum_x += self.y_train[j] * self.alphas[j] * K[j, i]
            y_pred[i] = sum_x

        print(y_pred)
        return y_pred

    def predict(self, X):
        scores = self.decision_function(X)
        median_score = np.median(scores)
        #return [1 if sc > median_score else -1 for sc in scores]
        return np.sign(self.decision_function(X))


    def calc_k(self, X_train, X_test):
        n_rows = X_train.shape[0]
        n_cols = X_test.shape[0]
        K = np.zeros((n_rows, n_cols))
        for i in xrange(n_rows):
            for j in xrange(n_cols):
                K[i, j] = self.kernel.f(X_train, X_test, i, j)
        return K

class Kernel(object):
    def __init__(self, ktype='dot'):
        self.type = ktype

    def f(self, x, y, i, j):
        if self.type == 'dot':
            return np.dot(x[i], y[j])
        if self.type == 'gaussian':
            sigma = 1
            return np.exp(-np.linalg.norm(x[i]-y[j], 2)**2/(2.*sigma**2))

        else:
            return np.sqrt(np.sum((x[i] - y[j]) ** 2))

