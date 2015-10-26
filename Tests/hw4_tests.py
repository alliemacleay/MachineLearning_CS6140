__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.Homeworks.hw4 as hw4
import CS6140_A_MacLeay.utils.Adaboost as adab
import CS6140_A_MacLeay.utils as utils
import numpy as np

def UnitTests():
    #AdaboostErrorTest()
    #AdaboostWrongTest()
    changeWeight()

def AdaboostErrorTest():
    spamData = get_test_always_right()
    adaboost_run(spamData)

def AdaboostWrongTest():
    d = get_test_always_wrong()
    adaboost_run(d)

def changeWeight():
    d = get_test_half_right()
    adaboost_run(d, 3)


def adaboost_run(data, num_rounds=2):
    adaboost = adab.Adaboost(num_rounds)
    adaboost.run(data)
    adaboost.print_stats()

def get_test_always_right():
    d = np.ones(shape=(100, 2))
    return d

def get_test_always_wrong():
    d = np.zeros(shape=(100, 2))
    return d

def get_test_half_right():
    d = np.ones(shape=(100, 2))
    for i in range(len(d)/2):
        d[i][-1] = 0
    #print d
    return d


def testData():
    # Create the dataset
    rng = np.random.RandomState(1)
    X = np.linspace(0, 6, 100)[:, np.newaxis]
    y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
    return X, y

if __name__ == '__main__':
    #hw4.q1()
    UnitTests()
