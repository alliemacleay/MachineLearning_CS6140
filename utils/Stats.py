__author__ = 'Allison MacLeay'
import numpy as np
import pandas as pd
import sys
#import CS6140_A_MacLeay.utils as utils
from CS6140_A_MacLeay import utils

from sklearn.cross_validation import KFold


def compute_accuracy(tp, tn, fp, fn):
    return float(tp+tn)/(tn+tp+fp+fn)

def compute_ACC(predicted, observed):
    [tp, tn, fp, fn] = get_performance_stats(predicted, observed)
    return compute_accuracy(tp, tn, fp, fn)

def compute_MSE_arrays(predicted, observed):
    T = len(observed)
    if T != len(predicted):
        print 'WARNING: len(o) {} is not equal to len(p) {}'.format(T, len(predicted))
    observed = list(observed)
    sig = 0
    if T == 0:
        return 0
    for i, p in enumerate(predicted):
        sig += (p-observed[i])**2
    return float(sig)/T

def compute_MSE(predicted, observed):
    """ predicted is scalar and observed as array"""
    if len(observed) == 0:
        return 0
    err = 0
    for o in observed:
        err += (predicted - o)**2/predicted
    return err/len(observed)

def compute_combined_MSE(A, B):
    """ """
    if len(A) == 0:
        return 0
    muA = utils.average(A)
    muB = utils.average(B)
    if muA == 0:
        muA += .000000001
    if muB == 0:
        muB += .000000001
    total = 0
    total += compute_MSE(muA, A)
    total += compute_MSE(muB, B)

    return total

def mse(df, col):
    mu = utils.average(df[col])
    sig = 0
    for i in df[col]:
        sig += (i-mu)**2
    return float(sig)/len(df[col])

def calculate_chisq_error(pred, truth):
    """ (E-O)^2/E """
    i = 0
    err = 0
    for p in pred:
        t = truth[i]
        err += (t - p)**2/t
        i += 1
    return err/len(truth)

def calculate_binary_error(pred, truth):
    total = len(pred)
    positves_predicted = sum(pred)
    true_positive = sum(np.logical_and(pred, truth))
    true_negative = sum(np.logical_and(np.logical_not(pred), np.logical_not(truth)))
    correct = true_negative + true_positive
    error = float(total - correct)/total
    print 'Total: %s' % total
    print 'True Positive: %s' % true_positive
    print 'True Negative: %s' % true_negative
    print 'Positives Predicted: %s' % positves_predicted
    print 'Correctly Predicted: %s' % correct
    print 'Error: %s' % error
    return error


def binary_info_gain(df, feature, y):
    """
    :param df: input dataframe
    :param feature: column to investigate
    :param y: column to predict
    :return: information gain from binary feature column
    """
    return sum(np.logical_and(df[feature], df[y]))/len(df[feature])


def get_performance_stats(truth, predict):
    print 'len: ' + str(len(truth)) + ' : ' + str(len(predict))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(truth)):
        if predict[i] == 1:
            if truth[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if truth[i] == 0:
                tn += 1
            else:
                fn += 1
    return [tp, tn, fp, fn]




def binary_probability(df, y):
    """ probability for features with 0 and 1 values """
    if len(df[y]) is 0:
        prob = 0
    else:
        prob = sum(df[y])/len(df[y])
    return prob

def binary_entropy(df, y):
    """ entropy for features with 0 and 1 values """
    return binary_probability(df, y) * len(df)


def least_squares(df, y=None):
    """ Option to pass in array rather than dataframe and column y """
    if type(df) is not list and y is not None:
        df = list(df[y])
    if len(df) == 0:
        return 0
    mu = sum(df)/len(df)
    sigma = 0
    for i in range(0, len(df)):
        sigma += (df[i] - mu)**2
    return sigma/2


def binary_error(df, y, predicted):
    error = binary_probability(df, y)
    if predicted is 1:
        error = 1 - error
    return error


def get_linreg_w(X, Y):
    """ X: dataframe of x1, x2, x..., xn
    Y: array of y
    return: w as matrix """
    print X
    Xt = X.transpose()
    #w_den = np.mat(Xt) * np.mat(X)
    w_den = np.dot(Xt, X)
    #w_pre = np.mat(utils.matrix_inverse(w_den)) * np.mat(Xt)
    #print w_den
    w_pre = np.dot(utils.matrix_inverse(w_den), Xt)
    #w = np.mat(list(Y)) * np.mat(w_pre)
    w = np.dot(w_pre, Y)
    return w

def get_linridge_w(X_uncentered, Y, learning_rate):
    """ Linear ridge
    X: dataframe of x1, x2, x..., xn
    Y: array of y
    return: w as matrix """
    Xdict = {}
    for index, row in X_uncentered.iterrows():
        print row
        x_bar = row.mean()
        Xdict[index] = []
        for col in row:
            Xdict[index].append(col - x_bar)

    Xt = pd.DataFrame(Xdict)

    X = Xt.transpose()
    I = np.identity(X.shape[1])
    w_den = np.dot(Xt, X) + np.dot(learning_rate, I)
    w_pre = np.dot(utils.matrix_inverse(w_den), Xt)
    w = np.dot(w_pre, Y)
    return w

def linear_regression_points(X_old, Y):
    #print Y
    Y_fit = []
    X = pd.DataFrame(X_old.copy())
    X['b'] = np.ones(len(X))
    w = get_linreg_w(X, Y)
    print 'w is: '
    print w
    for i, col in enumerate(X.columns):
        Y_fit.append(w[i] * X[col])
    return Y_fit

def linear_ridge_points(X_old, Y, learning_rate=.05):
    #print Y
    Y_fit = []
    X = pd.DataFrame(X_old.copy())
    X['b'] = np.ones(len(X))
    w = get_linridge_w(X, Y, learning_rate)
    print 'w is: '
    print w
    for i, col in enumerate(X.columns):
        Y_fit.append(w[i] * X[col])
    return Y_fit

def k_folds(df, k):
    """ k folds for hw1 prob 2"""
    number = np.floor(len(df)/k)
    print number
    folds = []
    #for i in range(0, k):
    #    folds.append(df.sample(number, replace=False))
    kf = KFold(len(df), n_folds=k)
    return kf

def get_error(predict, truth, is_binary):
    if is_binary:
        error = compute_ACC(predict, truth)
    else:
        error = compute_MSE_arrays(predict, truth)
    return error

    ######################
    """
    for train, test in kf:
        print 'test and train'
        print test
        print train
    return folds
    """

def log_likelihood(array):
    p = 1
    for i in range(0,len(array)):
        p *= array[i]
    return np.log(p)

def init_w(size):
    df = pd.DataFrame(np.random.random(size))
    return df.reindex()















