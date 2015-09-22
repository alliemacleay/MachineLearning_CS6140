__author__ = 'Allison MacLeay'
import numpy as np
import pandas as pd
import sys
import utils

from sklearn.cross_validation import KFold


def compute_accuracy(tp, tn, fp, fn):
    return float(tp+tn)/(tn+tp+fp+fn)

def compute_ACC(predicted, observed):
    #TO DO - fix this implementation
    chi_sq_errors = calculate_chisq_error(predicted, observed)
    return chi_sq_errors

def compute_MSE_arrays(predicted, observed):
    T = len(observed)
    observed = list(observed)
    sig = 0
    if T == 0:
        return 0
    for i, p in enumerate(predicted):
        sig += (p-observed[i])**2
    return sig/T

def compute_MSE(predicted, observed):
    """ predicted is scalar and observed as array"""
    if len(observed) == 0:
        return 0
    err = 0
    for o in observed:
        err += (predicted - o)**2/predicted
    return err/len(observed)

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

def k_folds(df, k):
    """ k folds for hw1 prob 2"""
    number = np.floor(len(df)/k)
    print number
    folds = []
    #for i in range(0, k):
    #    folds.append(df.sample(number, replace=False))
    kf = KFold(len(df), n_folds=k)
    return kf

    ######################
    """
    for train, test in kf:
        print 'test and train'
        print test
        print train
    return folds
    """

















