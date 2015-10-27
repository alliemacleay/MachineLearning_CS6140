__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.Homeworks.hw4 as hw4
import CS6140_A_MacLeay.utils.Adaboost as adab
import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.Homeworks.HW4 as hw4
import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import numpy as np

def UnitTests():
    #AdaboostErrorTest()
    #AdaboostWrongTest()
    #TestAbstract()
    #changeWeight()
    TreeTest()

def TestAbstract():
    d = get_test_always_right()
    ada = adab.AdaboostOptimal(1)
    ada.run(d)
    ada.print_stats()

def TreeTest():
    spam = spamData()
    truth, f_data = adab.split_truth_from_data(spam)
    tree = hw4.TreeOptimal()
    tree.fit(f_data, truth)


def AdaboostErrorTest():
    print 'Always right'
    spamData = get_test_always_right()
    adaboost_run(spamData)

def AdaboostWrongTest():
    print 'Always wrong'
    d = get_test_always_wrong()
    adaboost_run(d)

def changeWeight():
    d = get_test_half_right()
    adaboost_run(d, 3)


def adaboost_run(data, num_rounds=2):
    adaboost = adab.AdaboostOptimal(num_rounds)
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

def spamData():
    return hw3.pandas_to_data(hw3.load_and_normalize_spambase())

if __name__ == '__main__':
    #hw4.q1()
    UnitTests()
