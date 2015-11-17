import CS6140_A_MacLeay.Homeworks.HW6 as hw6u
import CS6140_A_MacLeay.Homeworks.HW3 as hw3u
import CS6140_A_MacLeay.Homeworks.HW4 as hw4u
import CS6140_A_MacLeay.Homeworks.HW4.data_load as dl
import CS6140_A_MacLeay.utils as utils
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

""" Homework 6
    11/16/15
"""
__author__ = 'Allison MacLeay'


def q1a():
    """SVM on Spam Data
    length train: 4140 length test 461
    train acc: 0.806763285024 test acc: 0.819956616052
    """
    data = utils.pandas_to_data(utils.load_and_normalize_spam_data())
    svm_q1(data)


def q1b():
    """SVM on haar dataset
        multiclass!

    """
    mnsize = 100
    data = utils.pandas_to_data(hw6u.load_mnist_features(mnsize))
    k = 10
    all_folds = hw3u.partition_folds(data, k)
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train)
    y_test, X_test = hw4u.split_truth_from_data(kf_test)
    clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
    y_pred = clf.predict(X)
    print 'train acc: {} test acc: {}'.format(accuracy_score(y, y_pred), accuracy_score(y_test, clf.predict(X_test)))



def svm_q1(data):
    k = 10
    all_folds = hw3u.partition_folds(data, k)
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train)
    y_test, X_test = hw4u.split_truth_from_data(kf_test)
    print 'length train: {} length test {}'.format(len(X), len(X_test))
    clf = svm.SVC()
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    print 'train acc: {} test acc: {}'.format(accuracy_score(y, clf.predict(X)), accuracy_score(y_test, clf.predict(X_test)))



def q2():
    pass


def q3():
    pass


def q4():
    pass


def q5():
    pass



