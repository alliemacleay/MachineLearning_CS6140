__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.utils as utils
import pandas as pd
import CS6140_A_MacLeay.utils.Stats as mystats
import numpy as np
from CS6140_A_MacLeay.utils.Stats import multivariate_normal
#from scipy.stats import multivariate_normal # for checking

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

def get_covar(data):
    arr = np.array(data).T
    #return np.cov(transpose_array(data))
    return np.cov(arr)
    #return np.corrcoef(arr)

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

    def train2(self, X, mus, covar_matrix, label):
        """ """
        #TODO train subsets together
        tmp = X[:]
        if type(X) is list:
            X = np.matrix(X)

        prob = []
        x_less = X - mus
        # this is called for each class outside
        for r, row in enumerate(x_less):
            row = np.asarray(x_less[r]).ravel()
            prob.append(self.multivariate_normal(covar_matrix, row, mus=mus))
        # Alternate way to do this found below.
        # http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
        X = tmp[:]
        self.update_prob(label, prob)

    def train(self, X, covar_matrix, y):
        """ """
        #TODO train subsets together
        self.prob = {}
        mus = {}
        #TODO my mus passed in are wrong - should be mus from total set
        for label in [0, 1]:
            self.prob[label] = [0 for _ in range(len(X))]
            mus[label] = get_mus(get_sub_at_value(X, y, label))
        mus['X'] = get_mus(X)
        for label in [0, 1, 'X']:
            prob = []
            #sub_data, sub_truth, sub_indeces = get_data_truth(X, y, mus['X'], label)
            x_less = [np.asarray(X[xi]) - mus[label] for xi in range(len(X))]
            # this is called for each class outside
            for r, row in enumerate(x_less):
                row = np.asarray(x_less[r]).ravel()
                prob.append(self.multivariate_normal(covar_matrix, row, mus=mus[label]))

            self.prob[label] = prob
        # now we have prob = [0: prob_x0_rows, 1:prob_x1_rows, 'X':prob_x_rows]

        # Alternate way to do this found below.
        # http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
        #self.update_prob(label, prob)

    def aggregate_model(self, models):
        for gda in models:
            pass

    def multivariate_normal(self, covar_matrix, x_less, alpha=1, mus=[]):
        """
        :param d: number of rows in X
        :param covar_matrix:
        :param x_less: X - u , u is a vector of mu
        :return:
        """
        return multivariate_normal(covar_matrix, x_less, alpha=1)

    def update_prob(self, label, prob):
        if self.prob is None:
            self.prob = {label: prob}
        elif label not in self.prob.keys():
            self.prob[label] = prob

    def predict(self, data):
        predicted = []
        lprob = {x: 1 for x in self.prob.keys()}
        #TODO - Do I use Gaussian process instead?
        for r in range(len(data)):
            max_prob_label = None
            for label in [0, 1]:
                if max_prob_label is None:
                    max_prob_label = [self.prob[label][r], label]
                else:
                    if self.prob[label][r] > max_prob_label[0]:
                        max_prob_label = [self.prob[label][r], label]
            predicted.append(max_prob_label[1])
        return predicted

    def normalize_probabilities(self):
        for i in range(len(self.prob[0])):
            Z = self.prob[0] + self.prob[1]
            self.prob[0] = self.prob[0] / Z
            self.prob[1] = self.prob[1] / Z


def partition_folds(data, k):
    #TODO - is this wrong??
    if k == 1:
        return [data]
    if len(data) > k:
        array = [[] for _ in range(k)]
    else:
        array = [[] for _ in range(len(data))]
    #array = []
    for i in range(len(data)):
        array[i % k].append(data[i])
    return array

def get_accuracy(predict, truth):
    right = 0
    for i in range(len(predict)):
        if predict[i] == truth[i]:
            right += 1
    return float(right)/len(predict)


def get_data_and_mus(spamData):
    truth_rows = transpose_array(spamData)[-1]  # truth is by row
    data_rows = transpose_array(transpose_array(spamData)[:-1])  # data is by column
    data_mus = get_mus(data_rows)
    y_mu = utils.average(truth_rows)
    return truth_rows, data_rows, data_mus, y_mu

def get_data_truth(data_rows, truth_rows, data_mus, label):
    data = []
    mus = []
    indeces = []
    truth = []
    for i in range(len(truth_rows)):
        if truth_rows[i] == label:
            data.append(data_rows[i])
            indeces.append(i)
            truth.append(truth_rows[i])

    return data, truth, indeces

def get_std_dev(data):
    std_dev = []
    by_col = transpose_array(data)
    for col in by_col:
        std_dev.append(np.std(col))
    return std_dev

def univariate_normal(data, std_dev, mus, prob_y, alpha=1):
        """
        :row: one row
        :param std_dev:  array by col
        :param mus: array by col
        :return: probability
        """
        row_probability = []
        # 1/(std_dev * sqrt(2*pi) ) exp( -1 * (x-mu)**2 / 2 * std_dev**2 )
        epsilon = 1. * alpha/len(std_dev)
        prob_const = 1./ np.sqrt(2 * np.pi)
        for row in data: # for each row
            prob = 1
            for j in range(len(row)):
                std_devj = std_dev[j] + epsilon
                xj = row[j]
                epow = -1 * (xj - mus[j])**2 / (2 * std_devj**2)
                probj = prob_const * (1.0/std_devj) * np.exp(epow)
                prob = probj * prob
                # >>> probj
                #5.9398401853736429
                # >>> scipy.stats.norm(mus[j], std_devj).pdf(xj)
                #5.9398401853736429
            row_probability.append(prob * prob_y)
        return row_probability

def bins_per_column(data_cols, cutoffs):
    column_prob = []
    num_bins = len(cutoffs)
    for c in range(len(data_cols)):
        prob = [0 for _ in range(num_bins)]
        counts = classify(data_cols[c], cutoffs)
        # add all bin counts for this column
        for xbin_i in range(len(counts)):
            prob[xbin_i] += counts[xbin_i]
        prob = [float(prob[i]) / len(data_cols[c]) for i in range(num_bins)]
        column_prob.append(prob)
    return column_prob

def bins_per_column_by_col(data_cols, cutoffsc):
    column_prob = []
    num_bins = len(cutoffsc)
    for c in range(len(data_cols)):
        prob = [0 for _ in range(num_bins)]
        counts = classify(data_cols[c], cutoffsc[c])
        # add all bin counts for this column
        for xbin_i in range(len(counts)):
            prob[xbin_i] += counts[xbin_i]
        prob = [float(prob[i]) / len(data_cols[c]) for i in range(num_bins)]
        column_prob.append(prob)
    return column_prob

def classify(row, cutoffs):
    """ Classify a row for bins and return counts """
    xbin = [0 for _ in range(len(cutoffs))]
    for j in range(len(row)):
       xbin[classify_x(row[j], cutoffs)] += 1
    return xbin

def classify_x(x, cutoffs):
    """ Classify a data point for bins """
    bins = len(cutoffs)
    binlabel = 1
    # first entry is minimum.  skip it
    while binlabel < bins and x >= cutoffs[binlabel]:
        binlabel += 1  # increment until row[j] is greater than binlabel
    return binlabel - 1





