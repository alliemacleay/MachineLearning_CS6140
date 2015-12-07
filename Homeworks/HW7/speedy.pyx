# -*- coding: utf-8 -*-

from __future__ import unicode_literals



import inspect
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
import numpy.linalg as la
import numpy as np
cimport numpy as np
from scipy.spatial.distance import cosine

#from libc.math cimport abs, cosine
from libc.stdlib cimport malloc, free
cimport cython


__author__ = 'Allison MacLeay'

class KNN(object):
    def __init__(self, n_neighbors=5, classifier=KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='minkowski', p=2)):
        self.k = n_neighbors
        self.classifier = classifier


    def predict(self, X_test, X, y):
        sciKNN = self.classifier
        sciKNN.fit(X, y)
        return sciKNN.predict(X_test)

class MyKNN(object):
    def __init__(self, n_neighbors=5, algorithm='brute', metric='minkowski', metric_params=None, p=2, cls_metric=np.mean, radius=None, density=False, outlier_label=None):
        self.n_neighbors = n_neighbors
        if metric == 'minkowski' and p == 2:
            self.kernel = Kernel('euclidean')
        else:
            self.kernel = Kernel()
        self.N = None
        self.cls_metric = cls_metric
        self.X_train = None
        self.y_train = None
        self.radius = radius
        self.density = density
        self.outlier_label = outlier_label
        self.outlier_index = None

    def fit(self, X, y):
        if type(X) is not np.ndarray:
            X = np.asarray(X)
        if type(y) is not np.ndarray:
            y = np.asarray(y)
        self.X_train = X
        self.y_train = y
        if self.outlier_label is not None:
            self.outlier_index = self.y_train.shape[0]
            self.y_train = np.append(self.y_train, self.outlier_label)


    def predict(self, X_test):
        return np.sign(self.decision_function(X_test))


    def decision_function(self, X_test):
        # Map to K
        print 'my predict {} {}'.format(self.n_neighbors, self.kernel.name())
        if type(X_test) is not np.ndarray:
            X_test = np.asarray(X_test)
        n_samples, n_features = X_test.shape
        n_samples_train = self.X_train.shape[0]
        K = np.zeros((n_samples, n_samples_train))
        for i, x_i in enumerate(X_test):  # x1
            for j, x_j in enumerate(self.X_train):  # x2
                K[i, j] = self.kernel.f(np.array(x_i), np.array(x_j))

        y_pred = np.zeros(X_test.shape[0])
        self.N = [[] for i in range(X_test.shape[0])]
        if self.radius is not None:
            #radius
            none_arr = []
            for i in range(K.shape[0]):
                tmp = zip(K[i, :], range(len(K[i, :])))
                for j in range(len(tmp)):
                    if tmp[j][0] < self.radius:
                        self.N[i].append(tmp[j])
                if len(self.N[i]) == 0:
                    self.N[i] = np.array([np.array([self.outlier_label, self.outlier_index])])
                    none_arr.append(i)
                self.N[i] = np.asarray(self.N[i])
            self.N = np.asarray(self.N)

            if len(none_arr) > 0:
                print '{} outliers'.format(len(none_arr))
                print none_arr

        elif self.density:
            ones = K[i, self.y_train > .5]
            zeros = K[i, self.y_train <= .5]
            n_ones = len(ones)
            n_zeros = len(zeros)
            p1 = float(len(ones)) / X_test.shape[0]
            pz_given_1 = float(np.sum(ones)) / n_ones
            pz_given_0 = float(np.sum(zeros)) / n_zeros
            #TODO this is so wrong
            #y_pred = [float(p1 * pz_given_1) / K[i, j] for i, j in xrange(K.shape[0]), xrange(K[0].shape[0])]
            y_pred = [] #float(p1 * pz_given_1) / K[i, j] for i, j in xrange(K.shape[0]), xrange(K[0].shape[0])]
        else:
            self.N = np.array([sorted(zip(K[i, :], range(len(K[i, :]))))[:self.n_neighbors] for i in range(K.shape[0])])
        if not self.density:
            for i in xrange(self.N.shape[0]):
                y_pred[i] = self.cls_metric([self.y_train[self.N[i][j][1]] for j in xrange(self.N[i].shape[0])])
        return y_pred

    # get_params needed for clone() in multiclass.py
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    # _get_param_names needed for clone() in multiclass.py
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        args, varargs, kw, default = inspect.getargspec(init)
        if varargs is not None:
            raise RuntimeError("scikit-learn estimators should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        # Remove 'self'
        # XXX: This is going to fail if the init is a staticmethod, but
        # who would do this?
        args.pop(0)
        args.sort()
        return args


class Kernel(object):
    def __init__(self, ktype='euclidean', sigma=1):
        self.sigma = sigma  # for Gaussian
        self.ktype = ktype
        self.f = None
        if ktype == 'euclidean' or ktype == 'minkowski':
            self.f = self.euclid
        if ktype == 'cosine':
            self.f = self.cosine
        if ktype == 'gaussian':
            self.f = self.gaussian
        if ktype == 'poly2':
            self.f = self.poly2

    def euclid(self, xi, xj, **kwargs):
        return np.sum([(xi[m]-xj[m]) ** 2 for m in range(xi.shape[0])])

    def cosine(self, xi, xj, **kwargs):
        return np.sum([cosine(x_i, x_j) for x_i, x_j in zip(xi, xj)])

    def gaussian(self, xi, xj, sigma=1, **kwargs):
        return np.sum([-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)) for x, y in zip (xi, xj)])

    def poly2(self, xi, xj, **kwargs):
        return np.dot(xi, xj) ** 2

    def name(self):
        return self.ktype


    def compute(self, xi, xj, **kwargs):
        return self.f(xi, xj)

def test():
    print 'in speedy test'
