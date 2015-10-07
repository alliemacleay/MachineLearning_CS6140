__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.utils.Tree as mytree
import CS6140_A_MacLeay.utils.Stats as mystats
import CS6140_A_MacLeay.utils.GradientDescent as gd
import CS6140_A_MacLeay.utils.NNet as nnet
import CS6140_A_MacLeay.utils.Perceptron as perc
import hw1
import numpy as np
import pandas as pd
import sys

def dec_or_reg_tree(df_train, df_test, Y):
    binary = utils.check_binary(df_train[Y])
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

def linear_reg(df_train, df_test, Y, ridge=False):
    binary = utils.check_binary(df_train[Y])
    columns = df_train.columns[:-1]
    if ridge:
        #Y_fit = mystats.linear_ridge_points(df_train[columns], df_train[Y])
        w = mystats.get_linridge_w(df_train[columns], df_train[Y], binary)
    else:
        #Y_fit = mystats.linear_regression_points(df_train[columns], df_train[Y])
        w = mystats.get_linreg_w(df_train[columns], df_train[Y])

    #print 'Y_fit'
    #print Y_fit
    #for i in range(0, len(Y_fit)):
    #    print str(Y_fit[i]) + ' -- ' + str(train['is_spam'][i])

    #col_MSE = {}
    #predict = []

    #for i, col in enumerate(columns):
    #    predict.append(Y_fit[i] + Y_fit[-1])
    predict = mystats.predict(df_train[columns], w, binary)
    #raw_input()
    error_train = mystats.get_error(predict, df_train[Y], binary)
    # TEST #
    columns = df_test.columns[:-1]
    if ridge:
        #Y_fit = mystats.linear_ridge_points(df_test[columns], df_test[Y])
        w = mystats.get_linridge_w(df_test[columns], df_test[Y], binary)
    else:
        #Y_fit = mystats.linear_regression_points(df_test[columns], df_test[Y])
        w = mystats.get_linreg_w(df_test[columns], df_test[Y])


    predict = mystats.predict(df_test[columns], w, binary)
    error_test = mystats.get_error(predict, df_test[Y], binary)
    return [error_train, error_test]


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
    model = gd.logistic_gradient(df_train, df_train[Y], .0001, max_iterations=5000)
    print model
    predict = gd.predict(df_train, model, binary)
    print predict
    error_train = mystats.get_error(predict, df_train[Y], binary)
    predict = gd.predict(df_test, model, binary)
    print predict
    error_test = mystats.get_error(predict, df_test[Y], binary)
    #TODO data in probabilities can't be negative
    error_train = 0
    error_test = 0
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
    #h_results.append(dec_or_reg_tree(h_train, h_test, 'MEDV')) # works
    #h_results.append(linear_reg(h_train, h_test, 'MEDV')) # works
    #h_results.append(linear_reg(h_train, h_test, 'MEDV', True)) # works
    #h_results.append(linear_gd(h_train, h_test, 'MEDV')) # works
    #h_results.append(logistic_gd(h_train, h_test, 'MEDV'))

    s_test, s_train = utils.split_test_and_train(utils.load_and_normalize_spam_data())
    #s_results.append(dec_or_reg_tree(s_train, s_test, 'is_spam')) # works
    #s_results.append(linear_reg(s_train, s_test, 'is_spam')) # works
    #s_results.append(linear_reg(s_train, s_test, 'is_spam', True)) # works
    #s_results.append(linear_gd(s_train, s_test, 'is_spam')) # works
    s_results.append(logistic_gd(s_train, s_test, 'is_spam'))
    print_results_1(s_results, h_results)


def q_2():
    """ Perceptron """
    test, train = utils.load_perceptron_data()
    print test[4]
    print train.head(5)
    model = perc.Perceptron(train, 4, .05, 100)
    #print model.model
    #print 'Training error'
    #model.print_score()
    #print 'Testing error'
    #model.print_score(model.get_score(model.get_predicted(test), test[4]))


def q_3():
    print 'Run Neural Network for question 3 in homework 2'
    nnet.run()


def homework2():
    q_1()
    #q_2()
    #q_3()


if __name__ =='__main__':
    homework2()



