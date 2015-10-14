__author__ = 'Allison MacLeay'

import numpy as np
import pandas as pd
from CS6140_A_MacLeay.utils import average, sigmoid, add_col, get_hw
import CS6140_A_MacLeay.utils.Stats as mystats
import sys
#import CS6140_A_MacLeay.Homeworks.HW3 as hw3u


def to_col_vec(x):
    return x.reshape((len(x), 1))

def gradient(X, Y, gd_lambda, descent=True, max_iterations=1000, binary=False):
    accepted = False
    iterations = 0

    if type(X) is pd.DataFrame:
        X = pandas_to_data(X)
    # add column of ones
    X = add_col(X, 1)
    w = [0 for _ in range(len(X[0]))]
    hrows = [0 for _ in range(len(X))]
    print type(list(Y))
    not_converged = len(X)
    while iterations < max_iterations and not_converged > 0:
        not_converged = len(X)
        iterations += 1
        for r in range(len(X)):  # r = i, c = j
            row = X[r]
            h = get_hw(row, w, binary)
            hrows[r] = h
            #TODO this doesn't seem right!
            #print 'values: {} {} '.format(list(Y)[r], h)
            if h-list(Y)[r] == 0:
                not_converged -= 1
            for c in range(len(row)):
                w[c] = w[c] - (gd_lambda * (h - list(Y)[r]) * row[c])

        debug_print(iterations, not_converged, hrows, Y)
    return w


def logistic_gradient(X, Y, gd_lambda, descent=True, epsilon_accepted=1e-6, max_iterations=10000000):
    accepted = False
    iterations = 0
    epsilon = 1
    X['b'] = np.ones(len(X))
    m = X.shape[1]  # number of cols
    print 'sh0: {} len(X): {}'.format(m, len(X))
    w_old = np.zeros(m)

    while not accepted:
        w_new = np.zeros(w_old.shape)
        for j in range(len(w_old)): # by col
            delta = 0.0
            for i in range(len(X)):  # by row
                delta += (Y.values[i] - sigmoid(np.dot(w_old, X.values[i]))) * X.values[i, j]
            w_new[j] = w_old[j] + gd_lambda * delta

        if np.any(np.isnan(w_new)):
            raise ValueError('NAN is found on iteration {}'.format(iterations))
        epsilon = sum(np.abs(w_new - w_old))/len(w_new)
        print 'epsilon: {}'.format(epsilon)
        print 'w:'
        print '{} iterations, w: {}'.format(iterations, w_new[:])
        w_old = w_new
        if epsilon < epsilon_accepted:
            accepted = True
        if iterations >= max_iterations:
            accepted = True
        iterations += 1
    return w_new

def predict_data(data, model, binary=True, logistic=True, theta=.5):
    # TODO  values are too low to be correct

    predict = []
    for i in range(len(data)):
        value = 0
        row = data[i]
        for j in range(len(row)):
            value += row[j] * model[j]
        if binary:
            if value > theta:
                predict.append(1)
            else:
                predict.append(0)
        else:
            predict.append(value)
    return predict

def predict(df, model, binary=False, logistic=False):
    if 'b' not in df.columns:
        df['b'] = 1
    if binary:
        cutoff = .5
    if binary and logistic:
        predictions = [sigmoid(x) for x in np.dot(df, model)]
    else:
        predictions = np.dot(df, model)
    if binary:
        for p in range(len(predictions)):
            if predictions[p] < cutoff:
                predictions[p] = 0
            else:
                predictions[p] = 1
    return predictions


def pandas_to_data(df):
    array = []
    for i in range(len(df)):  # row
        row = df.iloc[i]
        row_array = []
        for j in range(len(row)):
            row_array.append(row[j])
        array.append(row_array)
    return array

def debug_print(iters, nc, h, y):
    diffs = 0
    error = mystats.get_error(h, y, 0)
    for i, pred in enumerate(y):
        diffs += abs(pred - h[i])
    distance = float(diffs)/len(h)
    print "actual"
    print y[:5]
    print "predicted"
    print h[:5]
    print 'loop: {} num not converged: {} distance: {} MSE: {}'.format(iters, nc, distance, error)
