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


class Kernel(object):
    def __init__(self, ktype='euclidean', sigma=1):
        self.sigma = sigma  # for Gaussian
        self.ktype = ktype
        self.f = None
        if ktype == 'euclidean' or ktype == 'minkowski':
            self.f = self.euclid_fast
        if ktype == 'cosine':
            self.f = self.cosine
        if ktype == 'gaussian':
            self.f = self.gaussian
        if ktype == 'poly2':
            self.f = self.poly2

    def euclid(object self, np.ndarray[np.float_t, ndim=1] xi, np.ndarray[np.float_t, ndim=1] xj, **kwargs):
        return np.sqrt(np.sum([(xi[m]-xj[m]) ** 2 for m in range(xi.shape[0])]))

    def euclid_fast(object self, np.ndarray[np.float_t, ndim=2] X_test, np.ndarray[np.float_t, ndim=2] X_train, int i, int j):
        cdef float result = 0
        m = X_test.shape[1]
        for k in range(m):
            result += (X_test[i, k] - X_train[j, k]) ** 2
        return np.sqrt(result)

    def cosine(self, X, Xt, i, j):
        return np.dot(X[i], Xt[j].T) / (la.norm(X[i]) * la.norm(Xt[j]))
        #return cosine(X[i], Xt[j])

    def gaussian(self, X, Xt, i, j, sigma):
        return np.sum([-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)) for x, y in zip (X[i], Xt[j])])

    def poly2(self, X, Xt, i, j):
        return np.dot(X[i], Xt[j]) ** 2

    def name(self):
        return str(self.ktype + ' fast')


    def compute(self, xi, xj):
        return self.f(xi, xj)

def test():
    print 'in speedy test'

@cython.boundscheck(False) # turn of bounds-checking for entire function
def calc_K(object kernel, np.ndarray[np.float_t, ndim=2] X_test, np.ndarray[np.float_t, ndim=2] X_train):
    cdef int n_samples = X_test.shape[0]
    cdef int n_features = X_test.shape[1]
    cdef int n_samples_train = X_train.shape[0]
    cdef int i, j
    cdef np.ndarray[np.float_t, ndim=2] K = np.zeros((n_samples, n_samples_train))
    for i in range(n_samples):
        for j in range(n_samples_train):
            K[i, j] = kernel.f(X_test, X_train, i, j)
    return K


@cython.boundscheck(False)
def decision_function_radius(np.ndarray[np.float_t, ndim=2] K, np.ndarray[np.float_t, ndim=2] X_test, #np.ndarray[np.float_t, ndim=2] X_train,
                             np.ndarray[np.float_t, ndim=1] y_train, int n_neighbors, str kernel_name, #object kernel,
                             float radius, float outlier_label, int outlier_index,
                             cls_metric):
    # Map to K
    print '[Cython] my predict {} {}'.format(n_neighbors, kernel_name)

    cdef int i, j, ct_neighbors
    cdef int n_samples = X_test.shape[0]
    y_pred = np.zeros(n_samples)
    N = [[] for i in range(n_samples)]
    #radius
    none_arr = []
    for i in range(n_samples):
        ct_neighbors = 0
        for j in range(K.shape[1]):
            if K[i, j] < radius:
                ct_neighbors += 1
                N[i].append(j)
        if ct_neighbors == 0:
            N[i] = [outlier_index]
            none_arr.append(i)
        N[i] = np.asarray(N[i])
    N = np.asarray(N)

    if len(none_arr) > 0:
        print '{} outliers'.format(len(none_arr))
        print none_arr
    for i in xrange(N.shape[0]):
        y_pred[i] = cls_metric([y_train[N[i][j]] for j in xrange(N[i].shape[0])])

    return y_pred


"""
def decision_function_density(self, X_test):
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
"""
