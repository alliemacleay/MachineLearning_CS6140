from sklearn.metrics import roc_auc_score

__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.utils.Adaboost as adab
import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.Homeworks.HW4 as decTree
import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import CS6140_A_MacLeay.Homeworks.hw4 as hw4
import CS6140_A_MacLeay.Homeworks.HW4.plots as plt
import CS6140_A_MacLeay.Homeworks.HW4.data_load as dl
from sklearn import tree
from sklearn.datasets import load_iris, make_classification
import numpy as np
import os

def UnitTests():
    #AdaboostErrorTest()
    #AdaboostWrongTest()
    #TestAbstract()
    #changeWeight()
    TreeTest2()
    #TreeTest()
    #testPlot()
    #testBranchOptimal()
    #dataloads()

def dataloads():
    crx_data()
    dl.data_q4()

def testPlot():
    directory = '/Users/Admin/Dropbox/ML/MachineLearning_CS6140/CS6140_A_MacLeay/Homeworks'
    path= os.path.join(directory, 'test.pdf')
    plot = plt.Errors([[1,2,3]]).plot_all_errors(path)

def TestAbstract():
    d = get_test_always_right()
    ada = adab.AdaboostOptimal(1)
    ada.run(d)
    ada.print_stats()

def TreeTest():
    spamDat = spamData()
    k = 10
    all_folds = hw3.partition_folds(spamDat, k)
    num_in_fold = []
    err_in_fold = []
    for i in range(len(all_folds) - 1):
        spam = all_folds[i]
        num_in_fold.append(len(spam))
        truth, f_data = decTree.split_truth_from_data(spam)
        tree = decTree.TreeOptimal(max_depth=2)
        #tree = decTree.TreeRandom()
        tree.fit(f_data, truth)
        print 'Prediction...\n'
        predict = tree.predict(f_data)
        print predict
        print truth
        error = 1. - hw3.get_accuracy(predict, truth)
        err_in_fold.append(error)
        print 'Tree error is: {}'.format(error)
    spam = all_folds[k -1]
    truth, f_data = decTree.split_truth_from_data(spam)
    tree = decTree.TreeOptimal(max_depth=2)
    #tree = decTree.TreeRandom()
    tree.fit(f_data, truth)
    predict = tree.predict(f_data)
    error = 1. - hw3.get_accuracy(predict, truth)
    sum_training_err = 0
    for i in range(len(num_in_fold)):
        sum_training_err += err_in_fold[i]
        #sum_training_err += float(err_in_fold)/num_in_fold
    average_training_error = float(sum_training_err)/len(num_in_fold)
    print 'Average training error: {}\nAverage testing error: {}'.format(average_training_error, error)


def TreeTest2():
    iris = load_iris()


    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    print(roc_auc_score(y, clf.predict(X)))

    clf2 = decTree.TreeOptimal()
    clf2.fit(X, y)
    print(roc_auc_score(y, clf2.predict(X)))

def testBranchOptimal():
    data, truth = get_test_theta()
    branch = decTree.BranchOptimal(data, truth, np.ones(len(data)))
    theta = branch.choose_theta(data, truth)
    if theta != 5.5:
        print 'Optimal is broken! {} != 5.5'.format(theta)
    else:
        print 'Optimal works'


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

def get_test_theta():
    d = [10, 8, 8, 2, 2, 3, 0, 0, 0]
    y = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
    return d, y

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

def crx_data():
    dl.data_q3_crx()
    dl.data_q3_vote()


if __name__ == '__main__':
    #decTree.q1()
    hw4.q1()
    #UnitTests()
    #hw4.q2()
    #hw4.q3()
    #hw4.q4()
    #hw4.q6()
    #hw4.q7()
