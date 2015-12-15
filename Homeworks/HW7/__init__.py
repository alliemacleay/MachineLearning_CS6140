import inspect
import warnings
import collections
import cython
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
import numpy.linalg as la
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import os

import subprocess
subprocess.call(["cython", "-a", os.path.join(os.getcwd(), "CS6140_A_MacLeay/Homeworks/HW7/speedy.pyx")])

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)

import speedy



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
    def __init__(self, n_neighbors=5, algorithm='brute', metric='minkowski', metric_params=None, p=2, cls_metric=np.mean, radius=None, density=False, outlier_label=None, bandwidth=None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        if (metric == 'minkowski' and p == 2) or metric == 'euclidean':
            self.kernel = speedy.Kernel('euclidean')
        else:
            self.kernel = Kernel(ktype=metric)
        self.N = None
        self.cls_metric = cls_metric
        self.X_train = None
        self.y_train = None
        self.radius = radius
        self.density = density
        self.outlier_label = outlier_label
        self.outlier_index = None
        self.bandwidth = bandwidth # for density


    def fit(self, X, y):
        if type(X) is not np.ndarray:
            X = np.asarray(X)
        y = np.asarray(y, dtype=np.float)
        self.X_train = X
        self.y_train = y
        if self.outlier_label is not None:
            self.outlier_index = self.y_train.shape[0]
            self.y_train = np.append(self.y_train, self.outlier_label)


    def predict(self, X_test):
        dec = self.decision_function(X_test)
        dsz = len(dec)
        return [-1 if dec[i] <= 0 else 1 for i in range(dsz)]


    def decision_function(self, X_test):
        # Map to K
        print 'my predict {} {}'.format(self.n_neighbors, self.kernel.name())
        if type(X_test) is not np.ndarray:
            X_test = np.asarray(X_test)
        #K = speedy.calc_K(self.kernel, X_test, self.X_train)
        print('start kernel')
        K = calc_K(self.kernel, X_test, self.X_train)

        print 'my Kernel calculated'
        print K
        print K.shape
        y_pred = np.zeros(X_test.shape[0])
        if self.radius is not None:
            #radius
            return speedy.decision_function_radius(K, np.array(X_test), self.y_train, self.n_neighbors, self.kernel.name(),
                      float(self.radius), float(self.outlier_label), int(self.outlier_index), self.cls_metric)

        elif self.density:
            px_given_1 = np.zeros(K.shape[0])
            px_given_0 = np.zeros(K.shape[0])
            print set(self.y_train)
            p1 = float(np.sum(self.y_train > .5)) / self.y_train.shape[0]
            print(collections.Counter(self.y_train))
            print(p1)
            #p0_arr = np.zeros(K.shape[0])
            for i in range(K.shape[0]):
                #print('predict {}'.format(i))
                # k for each sample in test set i-test j-train

                ones = K[i, self.y_train > .5]
                zeros = K[i, self.y_train <= .5]
                print ones
                n_ones = len(ones)
                n_zeros = len(zeros)
                sum_ones = float(np.sum(ones))
                sum_zeros = float(np.sum(zeros))
                total = sum_ones + sum_zeros
                if total == 0:
                    px_given_1[i] = 0
                    px_given_0[i] = 0
                    continue
                px_given_1[i] = sum_ones / total
                px_given_0[i] = sum_zeros / total

            px1 = np.asarray([float(p1 * px_given_1[i]) for i in xrange(K.shape[0])])
            print(px1)

            px0 = np.asarray([float((1.0 - p1) * px_given_0[i]) for i in xrange(K.shape[0])])
            zs = [a + b for a, b in zip(px0, px1)]

            px1 /= zs
            px0 /= zs
            print(zip(px1, px0))
            y_pred = [1 if px1[i] > px0[i] else 0 for i in range(K.shape[0])]
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

def calc_K(kernel, X_test, X_train):
    n_samples = X_test.shape[0]
    n_samples_train = X_train.shape[0]
    K = np.zeros(shape=(n_samples, n_samples_train))
    for i in range(n_samples):
        for j in range(n_samples_train):
            K[i, j] = kernel.f(X_test, X_train, i, j)
    return K

class Kernel(object):
    def __init__(self, ktype='euclidean', sigma=1):
        self.sigma = sigma  # for Gaussian
        self.ktype = ktype
        self.f = None
        if ktype == 'euclidean' or ktype == 'minkowski':
            self.f = self.euclid
        if ktype == 'cosine':
            self.f = self.cosine
        if ktype == 'cosine_sci':
            self.f = self.cosine_sci
        if ktype == 'cosine_similarity':
            self.f = self.cosine_similarity
        if ktype == 'gaussian':
            self.f = self.gaussian
        if ktype == 'poly2':
            self.f = self.poly2
        if ktype == 'gaussian_sci':
            self.f = self.gaussian_sci
        if ktype == 'gaussian_density':
            self.f = self.gaussian_density
        if ktype == 'poly2_sci':
            self.f = self.poly2_sci

    def euclid(self, xi, xj, **kwargs):
        return np.sqrt(np.sum([(xi[m]-xj[m]) ** 2 for m in range(xi.shape[0])]))
        #return [np.sqrt(np.sum((xi[m] - xj[m]) **2)) for m in range(xi.shape[0])]

    def cosine(self, X, Xt, i, j):
        # X and Xt are vectors
        return 1-(np.dot(X[i], Xt[j].T) / (la.norm(X[i]) * la.norm(Xt[j])))  # equals cosine distance
        #return cosine(X[i], Xt[j])
        #return cosine_similarity(xi, xj)

    def cosine_similarity(self, X, Xt, i, j):
        return cosine_similarity(X[i], Xt[j])

    def cosine_sci(self, xi, xj):
         return 1-(np.dot(xi, xj.T) / (la.norm(xi) * la.norm(xj)))  # equals cosine distance


    def xxxgaussian(self, xi, xj, i=None, j=None, sigma=1, **kwargs):
        return np.sum([np.exp(-(la.norm(x-y) ** 2 / (2 * sigma ** 2))) for x, y in zip (xi, xj)])

    def gaussian(self, x, y, i=None, j=None, sigma=1, **kwargs):
        return np.exp(-(la.norm(x[i]-y[j]) ** 2 / (2 * sigma ** 2)))

    def gaussian_sci(self, xi, yj):
        sigma = 1
        return np.exp(-(la.norm(xi-yj) ** 2 / (2 * sigma ** 2)))

    def gaussian_density(self, x, y, i, j):
        deltaRow = x[i] - y[j]
        return np.exp(np.dot(deltaRow, deltaRow.T) / -(2**2))


    def poly2(self, x, y, i, j):
        return - np.dot(x[i], y[j]) ** 2
        #return np.sum[xi*yi+ xi**2 * yi**2 + 2*xi*yi for xi, yi in zip(x[i], y[i])]

    def poly2_sci(self, xi, xj, **kwargs):
        return - np.dot(xi, xj) ** 2
        #return np.sum[xi*yi+ xi**2 * yi**2 + 2*xi*yi for xi, yi in zip(x[i], y[i])]


    def name(self):
        return self.ktype


    def compute(self, xi, xj, **kwargs):
        return self.f(xi, xj)

def testCython():
    print 'out of speedy'
    speedy.test()







