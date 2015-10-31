__author__ = 'Allison MacLeay'
"""
Issues
---------
GD
Linear - My MSE's are unbelievably good but predicte value
    look good too
Logisitc - Either my classifier function is completely wrong
    or W is wrong.  I assume I could use the linear reg function
    but results are mostly less than .1 rather than .5 as
    expected

"""

import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.utils.Tree as mytree
import CS6140_A_MacLeay.utils.Stats as mystats
import CS6140_A_MacLeay.utils.GradientDescent as gd
import CS6140_A_MacLeay.utils.NNet as nnet
import CS6140_A_MacLeay.utils.Perceptron as perc
#import CS6140_A_MacLeay.Homeworks.HW4 as treeHW4
import HW4 as treeHW4
import hw1
import numpy as np
import pandas as pd
import sys

class Model_w():
    def __init__(self, w_array=None):
        self.w = w_array

    def update(self, w_array):
        if self.w == None:
            self.w = w_array
        else:
            if len(w_array) != len(self.w):
                print "ERROR!!!  Lengths are not the same"
            # Take the average
            sum = 0
            for i in range(len(w_array)):
                self.w[i] = float(w_array[i] + self.w[i])/2




def dec_or_reg_tree(df_train, df_test, Y):
    binary = utils.check_binary(df_train[Y])
    if binary:
        newtree = treeHW4.TreeOptimal()
        y = utils.pandas_to_data(df_train[Y])
        nondf_train = utils.pandas_to_data(df_train)
        nondf_test = utils.pandas_to_data(df_test)
        newtree.fit(nondf_train, y)
        predict = newtree.predict(nondf_train)
        error_train = mystats.get_error(predict, y, binary)

        y = utils.pandas_to_data(df_test[Y])
        predict = newtree.predict(nondf_test)
        error_test = mystats.get_error(predict, y)
    else:

        node = mytree.Node(np.ones(len(df_train)))
        hw1.branch_node(node, df_train, 5, Y)
        model = mytree.Tree(node)
        predict = model.predict_obj()
        error_train = mystats.get_error(predict, df_train[Y], binary)

        node.presence = np.ones(len(df_test))
        hw1.test_node(node, df_test, Y)
        test_tree = mytree.Tree(node)
        predict = test_tree.predict_obj()
        error_test = mystats.get_error(predict, df_test[Y], binary)
    return [error_train, error_test]


def linear_reg_errors(df_train, df_test, Y, ridge=False, sigmoid=False):
    binary = utils.check_binary(df_train[Y])
    error_train = linear_reg(df_train, Y, binary, ridge, sigmoid)
    error_test = linear_reg(df_test, Y, binary, ridge, sigmoid)
    return [error_train, error_test]

def linear_reg(df, Y, binary=False, ridge=False, sigmoid=False):
    means = []
    columns = [col for col in df.columns if (col != 'is_spam' and col != 'MEDV' and col != 'y')]
    if ridge:
        w = mystats.get_linridge_w(df[columns], df[Y], binary)
    else:
        for col in df.columns:
            mean = df[col].mean()
            means.append(mean)
            df[col] -= mean

        w = mystats.get_linreg_w(df[columns], df[Y])

    print('w:')
    print(w)
    predict = mystats.predict(df[columns], w, binary, means=means)
    error = mystats.get_error(predict, df[Y], binary)
    return error

def k_folds_linear_gd(df_test, df_train, Y):
    k = 10
    df_test = gd.pandas_to_data(df_test)
    k_folds = partition_folds(df_test, k)
    model = Model_w()
    theta = None
    for ki in range(k - 1):
        print 'k fold is {}'.format(k)
        data, truth = get_data_and_truth(k_folds[ki])
        binary = True
        model.update(gd.gradient(data, np.array(truth), .00001, max_iterations=5, binary=binary))
        print model.w
        if theta is None:
            theta, max_acc = get_best_theta(data, truth, model.w, binary, False)
        predict = gd.predict_data(data, model.w, binary, False, theta)
        error = mystats.get_error(predict, truth, binary)
        print 'Error for fold {} is {} with theta =  {}'.format(k, error, theta)
    test, truth = get_data_and_truth(k_folds[k-1])
    predict = gd.predict_data(test, model.w, binary, False, theta)
    test_error = mystats.get_error(predict, truth, binary)
    return [error, test_error]



def get_best_theta(data, truth, model, binary, logistic):
    best_theta = None
    max_acc = 0
    modmin = min(model)
    modmax = max(model)
    for theta_i in range(100):
        theta = modmin + float(theta_i)/(modmax - modmin)
        predict = gd.predict_data(data, model, binary, False, theta)
        acc = mystats.get_error(predict, truth, binary)
        if best_theta is None:
            best_theta = theta
            max_acc = acc
        elif acc > max_acc:
            best_theta = theta
            max_acc = acc
    return best_theta, max_acc


def linear_gd_error(df, Y):
    binary = utils.check_binary(df[Y])
    model = gd.gradient(df, df[Y], .00001, max_iterations=50)
    print model
    predict = gd.predict(df, model, binary)
    print predict
    error = mystats.get_error(predict, df_train[Y], binary)
    return error

def linear_gd(df_train, df_test, Y):
    """ linear gradient descent """
    binary = utils.check_binary(df_train[Y])
    model = gd.gradient(df_train, df_train[Y], .00001, max_iterations=50)
    print model
    predict = gd.predict(df_train, model, binary)
    print predict
    error_train = mystats.get_error(predict, df_train[Y], binary)
    predict = gd.predict(df_test, model, binary)
    print predict
    error_test = mystats.get_error(predict, df_test[Y], binary)
    return [error_train, error_test]

def logistic_gd(df_train, df_test, Y):
    """ logistic gradient descent """
    binary = utils.check_binary(df_train[Y])
    model = gd.logistic_gradient(df_train, df_train[Y], .1, max_iterations=5)
    print model
    predict = gd.predict(df_train, model, binary, True)
    print predict
    error_train = mystats.get_error(predict, df_train[Y], binary)
    predict = gd.predict(df_test, model, binary, True)
    print predict
    error_test = mystats.get_error(predict, df_test[Y], binary)
    return [error_train, error_test]

def print_results_1(spam, housing):
    j = 0
    line = '              '
    fields = ['Dec or Reg', 'Linear Reg', 'Linear Ridge', 'Linear Grad', 'Log Grad']
    for i in fields:
        line += '  {}  |'.format(i)
    print line
    line = '  Spam ACC train     '
    for i in spam:
        line += '  {}  |'.format(i[0])
    print line
    line = '       ACC test      '
    for i in spam:
        line += '  {}  |'.format(i[1])
    print line
    line = '  Housing MSE train  '
    for i in housing:
        line += '  {}  |'.format(i[0])
    print line
    line = '          MSE test   '
    for i in housing:
        line += '  {}  |'.format(i[1])
    print line






def q_1():
    h_test, h_train = utils.load_and_normalize_housing_set()
    h_results = []
    s_results = []
    h_results.append(dec_or_reg_tree(h_train, h_test, 'MEDV')) # MSE - 568 test- 448
    h_results.append(linear_reg_errors(h_train, h_test, 'MEDV')) # MSE - 27 test -14
    h_results.append(linear_reg_errors(h_train, h_test, 'MEDV', True)) # 24176 - 68289
    h_results.append(linear_gd(h_train, h_test, 'MEDV')) # works but MSE too low? .0022 - .0013
    #h_results.append(logistic_gd(h_train, h_test, 'MEDV'))  # 1.46e_13 - 1.17e+13

    s_test, s_train = utils.split_test_and_train(utils.load_and_normalize_spam_data())
    s_results.append(dec_or_reg_tree(s_train, s_test, 'is_spam')) # works .845 - .86
    s_results.append(linear_reg_errors(s_train, s_test, 'is_spam')) # works .8609 - .903
    s_results.append(linear_reg_errors(s_train, s_test, 'is_spam', True)) # works .8416 - .8543
    s_results.append(k_folds_linear_gd(s_train, s_test, 'is_spam')) # does not work .6114 - .6114
    s_results.append(logistic_gd(s_train, s_test, 'is_spam')) # returns perfect... 1- 1
    print_results_1(s_results, h_results)


def q_2():
    """ Perceptron """
    test, train = utils.load_perceptron_data()
    print test[4]
    print train.head(5)
    model = perc.Perceptron(train, 4, .05, 100)

def q_3():
    print 'Run Neural Network for question 3 in homework 2'
    nnet.run()


def homework2():
    q_1()
    #q_2()
    #q_3()


def partition_folds(data, k):
    if len(data) > k:
        array = [[] for _ in range(k)]
    else:
        array = [[] for _ in range(len(data))]
    for i in range(len(data)):
        array[i % 10].append(data[i])
    return array

def get_data_and_truth(data):
    print data
    x = []
    truth = []
    for r in range(len(data)):
        row = data[r]
        x.append(row[:-1])
        truth.append(row[-1])
    return x, truth

if __name__ =='__main__':
    homework2()





