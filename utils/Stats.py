__author__ = 'Allison MacLeay'
import numpy as np
import pandas as pd
import sys
#import CS6140_A_MacLeay.utils as utils
from CS6140_A_MacLeay import utils
from numpy.linalg import det, pinv, inv
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

def predict(df, model, binary=False, sigmoid=False, means=None):
    #if 'b' not in df.columns:
    #    df['b'] = 1
    #model = np.append(model,1)
    df['b'] = 1
    predictions = np.dot(df, model)
    if means is not None and len(means) == len(predictions):
        predictions = [predictions[i] + means[i] for i in range(len(predictions))]
    if binary:
        for p in range(len(predictions)):
            if predictions[p] < .5:
                predictions[p] = 0
            else:
                predictions[p] = 1
    return predictions

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
    return float(sum(np.logical_and(df[feature], df[y])))/len(df[feature])


def get_performance_stats(truth, predict):
    #print 'len: ' + str(len(truth)) + ' : ' + str(len(predict))
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
        prob = float(sum(df[y]))/len(df[y])
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
    mu = float(sum(df))/len(df)
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
    X['b'] = 1
    Xt = X.transpose()
    w_den = np.dot(Xt, X)
    w_pre = np.dot(utils.matrix_inverse(w_den), Xt)
    w = np.dot(w_pre, Y)
    del X['b']
    return w

def column_product(w, x):
    if type(w) is np.float64:
        w = list(w)
    sum = 0
    for i in range(len(w)):
        sum += w[i] * x[i]
    return sum

def dot_product_sanity(X, w):
    """X is a matrix, w is a vector"""
    result_vector = np.zeros(len(X.keys()))  # number of rows
    row_i = 0
    for row_k in X.keys():  # number of column
        row = X[row_k]
        result_vector[row_i] = column_product(row, w)
        row_i += 1
    return result_vector

def check_vector_equality(vec1, vec2):
    is_equal = True
    error_msg = []
    count_unequal = 0
    if len(vec1) != len(vec2):
        is_equal = False
        error_msg.append('rows are different sizes ({}, {})'.format(len(vec1), len(vec2)))
    else:
        for i in range(len(vec1)):
            if vec1[i] != vec2[i]:
                is_equal = False
                count_unequal += 1
    if is_equal:
        print 'Looks good!  Lengths are {}'.format(len(vec1))
    else:
        print '\n'.join(error_msg)
    return is_equal



def get_linridge_w(X_uncentered, Y, learning_rate):
    """ Linear ridge
    X: dataframe of x1, x2, x..., xn
    Y: array of y
    return: w as matrix """
    #TODO - add mean back in before predict

    X = X_uncentered

    X['b'] = 1
    Xt = X.transpose()

    I = np.identity(X.shape[1])
    w_den = np.dot(Xt, X) + np.dot(learning_rate, I)
    #w_den = np.cov(X) + np.dot(learning_rate, I)
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
    truth = np.array(truth)
    predict = np.array(predict)
    if is_binary:
        error = compute_ACC(predict, truth)
    else:
        error = compute_MSE_arrays(predict, truth)
    return error


def log_likelihood(array):
    p = 1
    for i in range(0,len(array)):
        p *= array[i]
    return np.log(p)

def init_w(size):
    df = pd.DataFrame(np.random.random(size))
    return df.reindex()

def summary(array):
    """ returns mean and variance"""
    return [utils.average(array), utils.variance(array, len(d))]


""" Added for lin ridge """
def pandas_to_data(df):
    array = []
    for i in range(len(df)):  # row
        row = df.iloc[i]
        row_array = []
        for j in range(len(row)):
            row_array.append(row[j])
        array.append(row_array)
    return array

def transpose_array(arr):
    tarry = []
    for i in range(len(arr)):
        if i == 0:
            for ix in range(len(arr[i])):
                tarry.append([])
        for j in range(len(arr[i])):
            tarry[j].append(arr[i][j])
    return tarry


def multivariate_normal(covar_matrix, x_less, alpha=1):
        """
        :param d: number of rows in X
        :param covar_matrix:
        :param x_less: X - u , u is a vector of mu
        :return:
        """
        covar_matrix = np.array(covar_matrix)
        x_less = utils.to_col_vec(np.asarray(x_less))
        epsilon = float(alpha * 1) / len(covar_matrix)
        set_diag_min(covar_matrix, epsilon)
        d = len(x_less)
        prob = float(1)/ ((2 * np.pi)**(float(d)/2))
        determinant = det(covar_matrix)
        if determinant == 0:
            print 'Determinant matrix cannot be singular'
        prob = prob * 1.0/(determinant**(float(1)/2))
        inverse = pinv(covar_matrix)
        dot = np.dot(np.dot(x_less.T, inverse), x_less)
        prob = prob * np.exp(-float(1)/2 * dot)
        #var = multivariate_normal(mean=mus, cov=determinant)
        return prob[0][0]


def set_diag_min(matrix, epsilon):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i==j and matrix[i][j] < epsilon:
                matrix[i][j] = epsilon















