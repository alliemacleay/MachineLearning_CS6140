__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.utils as utils
import pandas as pd
import CS6140_A_MacLeay.utils.Stats as mystats
import numpy as np
from numpy.linalg import det
from numpy.linalg import pinv, inv

def load_and_normalize_spambase():
    return utils.load_and_normalize_spam_data()

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

def get_mus(arr):
    """ Return averages of each vector
        Expects an array of arrays as input
    """
    trans = transpose_array(arr)  # to go by column
    mus = []
    for i in range(len(trans)):
        mus.append(utils.average(trans[i]))
    return mus

def calc_covar_X_Y(x, xmu, y, ymu):
    return (x-xmu)*(y-ymu)

def get_covar_X_Y(data, predict):
    """Data and predict are by rows
    """
    covar = []
    xmus = get_mus(data)
    ymu = utils.average(predict)
    for row in range(len(data)):
        covar.append([])
        y = predict[row]
        for i in range(len(data[row])):
            x = data[row][i]
            covar[row].append(calc_covar(x, xmus[i], y, ymu))
    return covar

def calc_covar(x, xmu):
    return (x-xmu)**2

def get_covar(data, p):
    return np.cov(transpose_array(data))

def mu_for_y(data, truth, value):
    """
    Returns averages for rows where y equals val
    :param data: by rows
    :param truth: by rows
    :param value: 0 or 1
    :return: array of averages by column
    """
    sub = get_sub_at_value(data, truth, value)
    return get_mus(sub)

def get_sub_at_value(data, truth, value):
    sub = []
    for i in range(len(truth)):
        if truth[i] == value:
            sub.append(data[i])
    return sub

def separate_X_and_y(data):
    y = []
    X = []
    for r in range(len(data)):
        y.append(data[r][-1])
        X.append(data[r][:-1])
    return X, y


class GDA():

    def __init__(self):
        self.prob = None
        self.predicted = None

    def train(self, X, mus, covar_matrix, label):
        """ """
        if type(X) is list:
            tmp = X[:]
            X = np.matrix(X)

        x_less = X - mus
        prob = self.gaussian_process(covar_matrix, x_less)
        # Alternate way to do this found below.
        # http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
        X = tmp[:]
        print len(prob)
        self.update_prob(label, prob)

    def gaussian_process(self, covar_matrix, x_less):
        """
        :param d: number of rows in X
        :param covar_matrix:
        :param x_less: X - u , u is a vector of mu
        :return:
        """
        d = len(x_less)
        prob = 1/ ((2 * np.pi)**(float(d)/2))
        prob = prob * 1/(det(covar_matrix)**(float(1)/2))
        dot = np.dot(np.dot(pinv(covar_matrix), x_less.T), x_less)
        prob = prob * np.exp(-float(1)/2 * dot)
        return prob

    def update_prob(self, label, prob):
        if self.prob is None:
            self.prob = {label: prob}
        if label not in self.prob.keys():
            self.prob[label] = prob
        else:
            #TODO - update
            self.prob[label] = prob

    def predict(self, data):
        if self.predicted is None:
            self.predicted = []
        lprob = {x: 1 for x in self.prob.keys()}
        #TODO - Do I use Gaussian process instead?
        for r in range(len(data)):
            for c in range(len(data[r])):
                for label in range(len(self.prob.keys())):
                    prob = self.prob[label][c]
                    # I already know this is wrong.  Prob is not returning
                    # the right size and values
                    lprob[label] = lprob[label] * prob
            max_prob_label = None
            for label in self.prob.keys():
                if max_prob_label is None:
                    max_prob_label = [self.prob[label], label]
                else:
                    if self.prob[label] > max_prob_label[0]:
                        max_prob_label = [self.prob[label], label]
            self.predicted.append(max_prob_label[0])


def partition_folds(data, k):
    if len(data) > k:
        array = [[] for _ in range(k)]
    else:
        array = [[] for _ in range(len(data))]
    for i in range(len(data)):
        array[i % 10].append(data[i])
    return array

def get_accuracy(predict, truth):
    right = 0
    for i in range(len(predict)):
        if predict[i] == truth[i]:
            right += 1
    return float(right)/len(predict)



