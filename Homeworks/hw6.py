from sklearn.metrics.scorer import accuracy_scorer
import CS6140_A_MacLeay.Homeworks.HW6 as hw6u
import CS6140_A_MacLeay.Homeworks.HW6.mysvm as mysvm
import CS6140_A_MacLeay.Homeworks.HW3 as hw3u
import CS6140_A_MacLeay.Homeworks.HW4 as hw4u
import CS6140_A_MacLeay.Homeworks.HW4.data_load as dl
import CS6140_A_MacLeay.utils as utils
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np

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
    svm_q1(data, svm.SVC())


def q1b():
    """SVM on haar dataset
       OneVsOneClassifier:
    Loading 9000 records from haar dataset
    Beginning analysis: (8100, 200)
    train acc: 0.91950617284 test acc: 0.81"""
    multiclassSVC(LinearSVC(random_state=0), 2000)

def multiclassSVC(classifier, sz=2000):

    mnsize = sz
    df = hw6u.load_mnist_features(mnsize)
    data = utils.pandas_to_data(df)
    k = 10
    all_folds = hw3u.partition_folds(data, k)
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train, replace_zeros=False)
    y, X = np.asarray(y), np.asarray(X)
    y_test, X_test = hw4u.split_truth_from_data(kf_test, replace_zeros=False)
    y_test, X_test = np.asarray(y_test), np.asarray(X_test)
    print 'Beginning analysis: {}'.format(X.shape)
    #clf = OneVsRestClassifier(classifier, n_jobs=4).fit(X, y)
    clf = OneVsOneClassifier(classifier).fit(X, y)
    #clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=10, random_state=0).fit(np.asarray(X), y)
    y_pred = clf.predict(X)
    print 'train acc: {} test acc: {}'.format(accuracy_score(fix_y(y_pred), fix_y(y)), accuracy_score(fix_y(y_test), fix_y(clf.predict(X_test))))
    print 'train acc: {} test acc: {}'.format(accuracy_score(fix_y(clf.predict(X)), fix_y(y)), accuracy_score(fix_y(y_test), fix_y(clf.predict(X_test))))



def svm_q1(data, classifier=svm.SVC()):
    k = 10
    all_folds = hw3u.partition_folds(data, k)
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train)
    y_test, X_test = hw4u.split_truth_from_data(kf_test)
    print 'length train: {} length test {}'.format(len(X), len(X_test))
    clf = classifier
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    print 'train acc: {} test acc: {}'.format(accuracy_score(fix_y(y), fix_y(clf.predict(X))), accuracy_score(fix_y(y_test), fix_y(clf.predict(X_test))))



def q2():
    data = utils.pandas_to_data(utils.load_and_normalize_spam_data())
    svm_q1(data, mysvm.SVC(mysvm.SMO, mysvm.Kernel('linear')))


def q3():
    multiclassSVC(mysvm.SVC(mysvm.SMO, mysvm.Kernel('linear')))


def q4():
    pass


def q5():
    pass

def fix_y(y_old):
    """ replace -1 with 0 """
    if y_old is np.float64 or y_old is np.isnan(y_old):
        pass
    if -1 in set(y_old):
        y = [0 if y_i==-1 else 1 for y_i in y_old]
    else:
        y = y_old
    return np.array(y)



if __name__ == "__main__":
    q2()
