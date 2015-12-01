# -*- coding: utf-8 -*-

"""\
(c) 2015 MGH Center for Integrated Diagnostics

"""

from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
cimport numpy as np

from libc.math cimport abs
from libc.stdlib cimport malloc, free
cimport cython


@cython.boundscheck(False) # turn of bounds-checking for entire function
def inner_2d(np.ndarray[np.float_t, ndim=2] X, int i, int j):
    cdef int k
    cdef float result
    result = 0
    for k in range(X.shape[1]):
            result += X[i, k] * X[j, k]
    return result


class Kernel(object):
    def __init__(self, ktype='linear', degree=3, coef0=0.):
        self.ktype = ktype
        self.degree = degree
        self.coef0 = coef0

    def f_fast(self, X, i, j):
        if self.ktype == 'linear':
            return inner_2d(X, i, j)
        elif self.ktype == 'poly':
            return self.f(X[i], X[j])
        else:
            return 'ERR: not implemented'

    def f(self, x, y):
        if self.ktype == 'linear':
            return np.inner(x, y)
        elif self.ktype == 'poly':
            return (self.coef0 + np.inner(x, y)) ** self.degree
        else:
            return 'ERR: not implemented'


@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef inline float bias(float b, float Ei, float yi, float yj, float aidiff, float ajdiff,
                       int i, int j, object kernel,
                       np.ndarray[np.float_t, ndim=2] K):
    # TODO: other kernels
    return b - Ei - yi * aidiff * K[i, i] - yj * ajdiff * K[i, j]


@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef inline float eval_f(int idx, np.ndarray[np.float_t, ndim=2] X, np.ndarray[np.float_t, ndim=1] y, object kernel, 
                         np.ndarray [np.float_t, ndim=1] alphas, float bias,
                         np.ndarray[np.float_t, ndim=2] K):
    cdef float sigma
    cdef int i
    sigma = bias
    for i in range(X.shape[0]):
        # TODO: other kernels
        sigma += alphas[i] * y[i] * K[idx, i]
    return sigma


cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b


@cython.boundscheck(False) # turn of bounds-checking for entire function
def myLagrangian(np.ndarray[np.float_t, ndim=2] X, np.ndarray[np.float_t, ndim=1] y,
                 object kernel, float c=1., float tolerance=1e-5, int maxiter=100):
        
        #if type(X) is not np.ndarray:
        #    X = np.asarray(X, dtype=float)
        #if type(y) is not np.ndarray:
        #    y = np.asarray(y, dtype=float)

        cdef float b, b1, b2, rho, aidiff, ajdiff
        cdef float n_changed
        cdef float fX
        cdef float Ei, Ej
        cdef float yE
        cdef np.ndarray[np.float_t, ndim=1] alphas
        cdef float passes
        cdef Py_ssize_t i, j
        cdef float L, H, alpha_i_old, alpha_j_old

        cdef np.ndarray[np.float_t, ndim=2] K
        K = np.zeros((X.shape[0], X.shape[0]))
        print('Computing kernel values...')
        for i in range(X.shape[0]):
                for j in range(X.shape[0]):
                        #K[i, j] = inner_2d(X, i, j)
                        K[i, j] = kernel.f_fast(X, i, j)
        print('Done!')

        b = 0.

        alphas = np.zeros(X.shape[0], dtype=float)  # number of samples
        passes = 0

        while passes < maxiter:
            n_changed = 0
            for i in range(X.shape[0]):
                fX = eval_f(i, X, y, kernel, alphas, b, K)
                Ei = fX - y[i]
                yE = y[i] * Ei
                if (yE < -tolerance and alphas[i] < c) or (yE > tolerance and alphas[i] > 0):
                    j = np.random.randint(0, X.shape[0]-1)
                    # ensure that j != i
                    if j >= i:
                        j += 1
                    Ej = eval_f(j, X, y, kernel, alphas, b, K) - y[j]
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]
                    if y[i] != y[j]:
                        L = float_max(0, alphas[j] - alphas[i])
                        H = float_min(c, c + alphas[j] - alphas[i])
                    else:
                        L = float_max(0, alphas[i] + alphas[j] - c)
                        H = float_min(c, alphas[i] + alphas[j])

                    if L == H:
                        continue
                    else:
                        rho = 2. * K[i, j] - K[i, i] - K[j, j]
                        if rho >= 0:
                            continue
                        else:
                            alphas[j] = alpha_j_old - y[j] * (Ei - Ej) / rho
                            if alphas[j] < L:
                                alphas[j] = L
                            elif alphas[j] > H:
                                alphas[j] = H
                            if abs(alphas[j] - alpha_j_old) < 1e-5:
                                continue
                            else:
                                alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alphas[j])
                            ajdiff = alphas[j] - alpha_j_old
                            aidiff = alphas[i] - alpha_i_old
                            b1 = bias(b, Ei, y[i], y[j], aidiff, ajdiff, i, j, kernel, K)
                            b2 = bias(b, Ej, y[j], y[i], ajdiff, aidiff, j, i, kernel, K)
                            if alphas[i] < c and alphas[i] > 0:
                                b = b1
                            elif alphas[j] < c and alphas[j] > 0:
                                b = b2
                            else:
                                b = (b1 + b2) / 2.0
                            n_changed += 1.
                            # end if
            # end for
            if n_changed == 0:
                passes +=1.
            else:
                passes = 0
            print('[Cython] %5s %5s %5s %5s' % ('pass', 'b','yE', 'numChanged'))
            print('[Cython] %i    %f  %f  %f' % (passes, b, yE, n_changed))
        return alphas, b
