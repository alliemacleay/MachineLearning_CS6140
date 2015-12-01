import numpy as np
from cvxpy import *
import cvxopt
import inspect
import warnings
from scipy.optimize import fsolve

import subprocess
subprocess.call(["cython","-a","Homeworks/HW6/superfast.pyx"])

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)

import superfast

__author__ = 'Allison MacLeay'

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


class SVC(object):
    def __init__(self, solver_type, kernel, bias=None):
        self.solver_type = solver_type
        self.kernel = kernel

        self.weights = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.bias = bias
        self.fit_status_ = None

    def fit(self, X, y):
        if type(y) is not np.ndarray:
            y = np.array(y, dtype=float)
        if 0 in set(y):
            y = self.fix_y(y)
        sample_weight = []
        solver_type = SMO(kernel=self.kernel)
        self.bias, self.weights, self.support_vectors, self.support_vector_labels = solver_type.solve(X, y)

    def fix_y(self, old_y):
        return np.array([1. if y_i == 1 else -1 for y_i in old_y], dtype=float)


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

    def predict(self, X, bias=None, weights=None, sv=None, svl=None, kernel=None):
        #print 'predict'
        y = np.ones(len(X))

        result = self.bias if bias is None else 0.
        weights = self.weights if weights is None else weights
        sv = self.support_vectors if sv is None else sv
        svl = self.support_vector_labels if svl is None else svl
        kernel = self.kernel if kernel is None else kernel
        for z_i, x_i, y_i in zip(weights,
                                 sv,
                                 svl):
            result += z_i * y_i * kernel.f(x_i, X)
        y = np.sign(result)
        if type(y) is not np.float64:
            y = np.asarray([0 if y_i < 0 else 1 for y_i in np.asarray(y)])
        return y

    def decision_function(self, X, dtype=float):
        """ binary estimator """
        result = self.bias
        weights = self.weights
        sv = self.support_vectors
        svl = self.support_vector_labels
        kernel = self.kernel
        for z_i, x_i, y_i in zip(weights,
                                 sv,
                                 svl):
            result += z_i * y_i * kernel.f(x_i, X)
        return result


class SMO(object):
    """ SMO is a solver """
    def __init__(self, kernel):
        self.model = None
        self.alpha = None
        self.sVectors = []
        self.sWeights = []
        self.b = None
        self.kernel = kernel

    def solve(self, X, y):
        if type(y) is not np.ndarray:
            y = np.array(y)
        if type(X) is not np.ndarray:
            X = np.array(X)
        n_pos = y[y>0].shape[0]
        n_neg = y.shape[0] - n_pos

        if n_pos == 0 or n_neg == 0:
            self.alpha = np.zeros(0, dtype=np.float_)
            self.b = 1. if n_pos > 0 else -1.
            self.sVectors = np.zeros((0, X.shape[1]), dtype=np.float_)
            self.sWeights = np.zeros(0, dtype=np.float_)
            return self.b, self.sWeights, self.sVectors, self.alpha

        alpha = np.zeros(y.shape[0], dtype=np.float) if self.alpha is None else self.alpha.copy()

        #real_lagrange_multipliers = Lagrangian(X, y, self.kernel)
        lagrange_multipliers, bias = superfast.myLagrangian(X, y, self.kernel, 1.0, 1e-2, 1)
        support_vector_indices = \
            lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
        if True not in support_vector_indices:
            print 'Warning: Empty lagrange multiplier vector.  Min is {} and max lagrangian is {}'.format(MIN_SUPPORT_VECTOR_MULTIPLIER, np.max(lagrange_multipliers))
        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
        # bias = y_k - \sum z_i y_i  K(x_k, x_i)
        # Find error from SVC for each sample
        computed_bias = np.mean(
            [y_k - SVC(SMO, self.kernel, 0.0).predict(x_k,
                weights=support_multipliers,
                sv=support_vectors,
                svl=support_vector_labels)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        print 'bias {} computed {}'.format(bias, computed_bias)
        return bias, support_multipliers, support_vectors, support_vector_labels,

def eval_f(x, X, y, kernel, alphas, bias):
    sigma = bias
    for i in range(X.shape[0]):
        sigma += np.sum(alphas[i] * y[i] * kernel.f(x, X[i]))
    return sigma

def bias(b, Ei, yi, yj, aidiff, ajdiff, Xi, Xj, kernel):
    return b - Ei - yi * aidiff * kernel.f(Xi, Xi) - yj * ajdiff * kernel.f(Xi, Xj)

def myLagrangian(X, y, kernel, c=1., tolerance=1e-5, maxiter=100):
        if type(X) is not np.ndarray:
            X = np.asarray(X, dtype=float)
        if type(y) is not np.ndarray:
            y = np.asarray(y, dtype=float)

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
                    j = np.random.randint(0, X.shape[0]-1, 1)[0]
                    # ensure that j != i
                    if j >= i:
                        j += 1
                    Ej = eval_f(X[j], X, y, kernel, alphas, b) - y[j]
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]
                    if y[i] != y[j]:
                        L = np.max([0, alphas[j] - alphas[i]])
                        H = np.min([c, c + alphas[j] - alphas[i]])
                    else:
                        L = np.max([0, alphas[i] + alphas[j] - c])
                        H = np.min([c, alphas[i] + alphas[j]])

                    if L == H:
                        continue
                    else:
                        rho = 2. * kernel.f(X[i], X[j]) - kernel.f(X[i], X[i]) - kernel.f(X[j], X[j])
                        if rho >= 0:
                            continue
                        else:
                            alphas[j] = alpha_j_old - y[j] * (Ei - Ej) / rho
                            if alphas[j] < L:
                                alphas[j] = L
                            elif alphas[j] > H:
                                alphas[j] = H
                            if np.abs(alphas[j] - alpha_j_old) < 1e-5:
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
                                b = np.mean([b1, b2])
                            n_changed += 1.
                            # end if
            # end for
            if n_changed == 0:
                passes +=1.
            else:
                passes = 0
            print '%5s %5s %5s %5s' % ('pass', 'b','yE', 'numChanged')
            print '%i    %f  %f  %f' % (passes, b, yE, n_changed)
        return alphas, b






def Lagrangian(X, y, kernel, c=10):
        # TODO replace with own
        # http://tullo.ch/articles/svm-py/
        if type(X) is not np.ndarray:
            X = np.asarray(X)
        n_samples, n_features = X.shape

        # Map to K
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):  # x1
            for j, x_j in enumerate(X):  # x2
                K[i, j] = kernel.f(x_i, x_j)

        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b



        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -a_i \leq 0
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.qp#cvxopt.solvers.qp
        return np.ravel(solution['x'])




