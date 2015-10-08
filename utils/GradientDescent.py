__author__ = 'Allison MacLeay'

import numpy as np
import pandas as pd
from CS6140_A_MacLeay.utils import average, sigmoid
import CS6140_A_MacLeay.utils.Stats as mystats


def to_col_vec(x):
    return x.reshape((len(x), 1))

def gradient(X, Y, gd_lambda, descent=True, epsilon_accepted=1e-6, max_iterations=1000):
    accepted = False
    iterations = 0
    epsilon = 1
    X['b'] = np.ones(len(X))
    x = X.transpose()
    m = X.shape[1]  # number of cols
    print 'sh0: {} len(X): {}'.format(m, len(X))
    w_old = np.zeros(m)
    while not accepted:
        diff = np.dot(np.dot(X.T, X), to_col_vec(w_old)) - np.dot(X.T, to_col_vec(Y))
        print 'diff {}'.format(diff)
        w_new = w_old - gd_lambda * diff.ravel()
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
