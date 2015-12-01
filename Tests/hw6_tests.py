from timeit import timeit

from sklearn.metrics.scorer import accuracy_scorer
import CS6140_A_MacLeay.Homeworks.hw6 as hw6
import CS6140_A_MacLeay.Homeworks.HW6 as hw6u
import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.Homeworks.HW6.mysvm as mysvm
import pandas as pd
import numpy as np
import os
import pickle
import cvxopt
import CS6140_A_MacLeay.Homeworks.HW6.mycvxopt as mycvxopt
from sklearn import svm

__author__ = 'Allison MacLeay'

def do_tests():
    #test_mnist_load()
    test_mnist_load_small()


def test_mnist_load_small():
    X = hw6u.load_mnist_features(10)
    print X.shape


def test_mnist_load():
    data = pd.read_csv('df_save_img_everything.csv')
    print len(data)
    X = utils.pandas_to_data(data)
    print X[0]

def test_SMO():
    X, y = testData(y_ones=True)
    print X
    print y
    smo = mysvm.SMO()
    smo.solve(X, y)


def test_cvxopt():
    mycvxopt.solvers().qp(0,0,0,0,0,0)
    path = '/Users/Admin/Dropbox/ml/MachineLearning_CS6140'
    with open(os.path.join(path, 'cvxopt.pkl'), 'rb') as f:
        arr = pickle.load(f)
    print 'pickle loaded'
    P = arr[0]
    q = arr[1]
    G = arr[2]
    h = arr[3]
    A = arr[4]
    b = arr[5]
    print 'input assigned'
    #     pcost       dcost       gap    pres   dres
    #0: -6.3339e+03 -5.5410e+05  2e+06  2e+00  2e-14
    #1:  5.8332e+02 -3.1277e+05  5e+05  2e-01  2e-14
    #2:  1.3585e+03 -1.3003e+05  2e+05  7e-02  2e-14
    #return np.ravel(solution['x'])
    with open(os.path.join(path, 'cvxopt_solution.pkl'), 'rb') as f:
        solution = pickle.load(f)
    print 'solution pickle loaded'

    mysolution = cvxopt.solvers.qp(P, q, G, h, A, b)
    print 'convex optimizer solved'
    if np.allclose(np.ravel(mysolution['x']), np.ravel(solution['x'])):
        print 'EQUAL!!!'
    else:
        print 'WROng!!!'

def test_mysvm():
    X, y = testData()
    X = np.concatenate([X, y.reshape((len(y), 1))], axis=1)
    X = [list(x.ravel()) for x in X]
    hw6.svm_q1(X, mysvm.SVC(mysvm.SMO, mysvm.Kernel('linear')))
    hw6.svm_q1(X, svm.SVC())


def testData(y_ones=False):
    # Create the dataset
    rng = np.random.RandomState(1)
    X = rng.rand(100, 2)
    y = np.asarray([0] * 50 + [1] * 50)
    X[y == 1] += 2.0
    return X, y


if __name__ == '__main__':
    #do_tests()
    #test_SMO()
    #test_cvxopt()
    timeit(test_mysvm, number=1)
    #hw6.q1a()
    #hw6.q1b()
    #hw6.q2()
    #hw6.q3()

