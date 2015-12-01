import numpy as np
import math
import cvxopt
from cvxopt import matrix, misc, blas

__author__ = 'Allison MacLeay'

# cvxopt/coneprog.py

class solvers(object):
    def __init__(self):
        pass

    def qp(self, P, q, G, h, A, b):
        round = 0
        print_round(None, 'pcost', 'dcost', 'gap', 'pres', 'dres')
        loop(P, q, G, h, A, b, round)


def takeStep(target, point, i1, i2):
    """
    target = y
    point = X
    """
    if i1 == i2:
        return 0
    alpha1 = Lagrangian(i1)
    y1 = target[i1]
    E1 = SVM(i1) - y1
    s = y1 * y2

def loop(P, q, G, h, A, b, round):
    #coneqp
    # P = np.outer(y, y) * K (373x373)
    # q = vector of -1 (373x1)
    # G = np.vstack( np.diag(np.ones(n_samples) * -1), np.diag(np.ones(n_samples)) )
    #     (746x373)
    # h = np.vstack( np.zeros(n_samples), np.ones(n_samples) * c )
    #     (746x1)
    # A = y
    # b = 0.0
    def fP(P, x, y, alpha=1., beta=0.):
        #base.symv
        return alpha * np.dot(P, x) + beta * y

    def fA(x, y, trans = 'N', alpha=1., beta=0.):
        #base.gemv
        # general matrix-vector product
        return cvxopt.gemv(x, y, trans, alpha, beta)
    dims = None
    kktsolver = 'chol2'
    if kktsolver == 'chol':
        factor = getattr(misc, 'kkt_chol')(G, dims, A)
    else:
        factor = getattr(misc, 'kkt_chol2')(G, dims, A)
    def kktsolver(W):
        return factor(W, P)

    f3 = kktsolver({'d': matrix(0.0, (0,1)), 'di':
        matrix(0.0, (0,1)), 'beta': [], 'v': [], 'r': [], 'rti': []})


    resx0 = max(1.0, math.sqrt(blas.dot(q,q)))  # 19.13
    resy0 = max(1.0, math.sqrt(blas.dot(b,b)))  # 1.0
    resz0 = max(1.0, misc.snrm2(h, dims))  # 193.13

    x = matrix(q)
    blas.scal(-1.0, x)
    y = matrix(b)
    f3(x, y, matrix(0.0, (0,1)))


    # dres = || P*x + q + A'*y || / resx0
    rx = matrix(q)
    fP(x, rx, beta = 1.0)
    pcost = 0.5 * (blas.dot(x, rx) + blas.dot(x, q))
    fA(y, rx, beta = 1.0, trans = 'T')
    dres = math.sqrt(blas.dot(rx, rx)) / resx0

    # pres = || A*x - b || / resy0
    ry = matrix(b)
    fA(x, ry, alpha = 1.0, beta = -1.0)
    pres = math.sqrt(blas.dot(ry, ry)) / resy0

    if pcost == 0.0:
        relgap = None
    else:
        relgap = 0.0

    cdim = G.size[0]
    x, y = matrix(q), matrix(b)
    s, z = matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1))
    pcost=-6.3339e+03
    dcost=-5.5410e+05
    gap= 2e+06
    pres=2e+00
    dres=2e-14
    print_round(round, pcost, dcost, gap, pres, dres)


def print_round(round, pcost, dcost, gap, pres, dres):
    if round is not None:
        # print numbers
        print '%2d: %.4e %.4e  %.0e  %.0e  %.0e ' % (round, pcost, dcost, gap, pres, dres)

    else:
        print '%2s %7s %11s %9s %7s %6s' % ('', pcost, dcost, gap, pres, dres)


def trash():

        def funcL(x, y, L):
            x = X[0]
            y = X[1]
            L = X[2]
            return kernel.f(x, y) * L
        def dfuncL(x, y, h=1e-3):  # h is the step size
            dLambda = np.zeros(len(X))
            for i in range(len(X)):
                dX = np.zeros(len(X))
                dX[i] = h
                dLambda[i] = (funcL(X + dX) - funcL(X - dX)) / (2*h)
            return dLambda

        H = fsolve(dfuncL, [1, 1, 0])
        L = fsolve(dfuncL, [-1, -1, 0])
