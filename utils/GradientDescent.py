__author__ = 'Allison MacLeay'

import numpy as np
import pandas as pd
from CS6140_A_MacLeay.utils import average
import CS6140_A_MacLeay.utils.Stats as mystats

def xgradient(X, Y, w_old, gd_lambda, descent=True, epsilon_accepted=5, max_iterations=None):
    """ Gradient Ascent or Descent
        defaults to gradient descent, epsilon = 5
    :return: w_new
    """
    sig = 0
    #TODO sig is NOT correct!
    for i, x in enumerate(X.iterrows()):
        #sig += np.dot(np.dot(x[1:], w_old) - Y[i], x[1:])
        sig += sum([(x[j]*w_old - Y[i])*x[j] for j in range(1, len(x))])
    #sig += np.dot(np.dot(X, w_old) - Y, X)

    if descent:
        w_new = w_old - gd_lambda * sig
    else:
         w_new = w_old + gd_lambda * sig
    e = np.abs(sum(w_new - w_old))
    print 'sig'
    print sig
    print 'w_old'
    print w_old
    print 'w_new'
    print w_new
    print 'epsilon= ' + str(e)
    keep_going = True
    if max_iterations is not None:
        max_iterations -= 1
        if max_iterations == 0:
            keep_going = False
    if e > epsilon_accepted and keep_going:
        w_new = gradient(X, Y, w_new, gd_lambda, descent, epsilon_accepted)
    return w_new

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
        #h = np.dot(X, w_old)
        #loss = [h[i] - list(Y)[i] for i in range(0, len(h))]
        #print 'loss {}'.format(loss)
        #MSE = sum([loss[i] ** 2 for i in range(0, len(loss))])/len(loss)
        #print 'MSE {}'.format(MSE)
        #diff = np.dot(X.transpose(), loss) / len(loss)
        diff = np.dot(np.dot(X.T, X), to_col_vec(w_old)) - np.dot(X.T, to_col_vec(Y))
        print 'diff {}'.format(diff)
        w_new = w_old - gd_lambda * diff.ravel()
        if np.any(np.isnan(w_new)):
            raise ValueError('NAN is found on iteration {}'.format(iterations))
        #epsilon = abs(sum(w_new - w_old)/len(w_new))
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
    x = X.transpose()
    m = X.shape[1]  # number of cols
    print 'sh0: {} len(X): {}'.format(m, len(X))
    w_old = pd.DataFrame(np.ones(m))
    #TODO these values can't be negative
    while not accepted:
        h = abs(1/(1 - np.exp(np.dot(X, w_old))))
        probabilities = [h[i] **(list(Y)[i]) * (1 - h[i] ** (1 - list(Y)[i])) for i in range(0, len(h))]
        log_likelihood = mystats.log_likelihood(probabilities)
        print 'log_likelihood {}'.format(log_likelihood)
        print list(Y) - h
        diff = gd_lambda * np.dot((Y - h), X)
        print 'diff {}'.format(diff)
        # gradient ascent
        w_new = w_old + gd_lambda * diff
        print w_new
        #epsilon = abs(sum(w_new - w_old)/len(w_new))
        epsilon = sum(w_new - w_old)/len(w_new)
        print 'iteration: {} epsilon: {}'.format(iterations, epsilon)
        #print w_new[:]
        w_old = w_new
        if epsilon < epsilon_accepted:
            accepted = True
        if iterations >= max_iterations:
            accepted = True
        iterations += 1
    return w_new


def get_slope_intercept(model, truth_set, binary=False):
    #print 'model'
    #print model
    truth = list(truth_set)
    slopes = np.zeros(len(model))
    intercepts = np.zeros(len(model))
    for i, x1 in enumerate(model):
        if i is 0:
            continue
        print i
        y1 = truth[i]
        y0 = truth[i-1]
        x0 = model[i-1]
        slopes[i] = (y1-y0)/(x1-x0)
        intercepts[i] = y1 - slopes[i]*x1
    print 'slopes'
    print average(slopes)
    print 'intercepts'
    print average(intercepts)
    return average(slopes), average(intercepts)


def predict(df, model, binary=False):
    if 'b' not in df.columns:
        df['b'] = 1
    predictions = np.dot(df, model)
    if binary:
        for p in range(len(predictions)):
            if predictions[p] < .5:
                predictions[p] = 0
            else:
                predictions[p] = 1
    return predictions


def xpredict(df, slopes, intercepts):
    for row in df.iterrows():
        print 'row: '
        print row
    predictions = []
    return predictions


def xxgradient(X, Y, gd_lambda, descent=True, epsilon_accepted=.005, max_iterations=10000000):
    accepted = False
    iterations = 0
    epsilon = 1
    X['b'] = np.ones(len(X))
    x = X.transpose()
    cols = X.shape[1]  # number of columns
    n = X.shape[0]  # number of samples
    print 'sh0: {} len(X): {}'.format(cols, len(X))
    t1 = np.random.random(n)
    t0 = np.zeros(n)

    # MSE
    J = sum([(t0 + t1*X[i] - Y[i])**2 for i in range(0, cols)])
    while not accepted:

        gradient0 = 1.0/cols * sum([t0 + t1*X[i] - Y[i] for i in range(0, cols)])
        gradient1 = 1.0/cols * sum([(t0 + t1*X[i] - Y[i])*X[i] for i in range(0, cols)])

        t0 = t0 - gd_lambda * gradient0
        t1 = t1 - gd_lambda * gradient1

        J_new = sum([(t0 + t1*X[i] - Y[i])**2 for i in range(0, cols)])

        epsilon = abs(J_new - J)
        J = J_new
        if epsilon < epsilon_accepted:
            accepted = True
        if iterations >= max_iterations:
            accepted = True
        iterations += 1
    return t0, t1


