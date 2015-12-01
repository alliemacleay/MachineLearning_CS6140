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
cdef inline float inner(np.ndarray[np.float_t, ndim=1] a, np.ndarray[np.float_t, ndim=1] b):
    cdef int i
    cdef float result
    result = 0
    for i in range(a.shape[0]):
            result += a[i] * b[i]
    return result



@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef inline float bias(float b, float Ei, float yi, float yj, float aidiff, float ajdiff,
                       np.ndarray[np.float_t, ndim=1] Xi, np.ndarray[np.float_t, ndim=1] Xj, object kernel):
    # TODO: other kernels
    return b - Ei - yi * aidiff * inner(Xi, Xi) - yj * ajdiff * inner(Xi, Xj)


@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef inline float eval_f(np.ndarray[np.float_t, ndim=1] x, np.ndarray[np.float_t, ndim=2] X, np.ndarray[np.float_t, ndim=1] y, object kernel, np.ndarray alphas, float bias):
    cdef float sigma
    cdef int i
    sigma = bias
    for i in range(X.shape[0]):
        # TODO: other kernels
        sigma += alphas[i] * y[i] * inner(x, X[i])
    return sigma


cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b


cdef float** convert_matrix(np.ndarray[np.float_t, ndim=2] Xin):
    cdef int i
    cdef float **X
    cdef float *x
    X = <float **>malloc(Xin.shape[0] * cython.sizeof(float))
    if X is NULL:
            raise MemoryError()

    for i in range(Xin.shape[0]):
            x = <float *>malloc(Xin.shape[1] * cython.sizeof(float))
            for j in range(Xin.shape[1]):
                    x[j] = Xin[i, j]
            X[i] = x
                    
    return X


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
        cdef float Ei
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
                        K[i, j] = inner(X[i], X[j])
        print('Done!')

        b = 0.

        alphas = np.zeros(X.shape[0], dtype=float)  # number of samples
        passes = 0

        while passes < maxiter:
            n_changed = 0
            for i in range(X.shape[0]):
                fX = eval_f(X[i], X, y, kernel, alphas, b)
                Ei = fX - y[i]
                yE = y[i] * Ei
                if (yE < -tolerance and alphas[i] < c) or (yE > tolerance and alphas[i] > 0):
                    j = np.random.randint(0, X.shape[0]-1)
                    # ensure that j != i
                    if j >= i:
                        j += 1
                    Ej = eval_f(X[j], X, y, kernel, alphas, b) - y[j]
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
                            b1 = bias(b, Ei, y[i], y[j], aidiff, ajdiff, X[i], X[j], kernel)
                            b2 = bias(b, Ej, y[j], y[i], ajdiff, aidiff, X[j], X[i], kernel)
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
