__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.utils.Tree as mytree
import CS6140_A_MacLeay.utils.Stats as mystats
import CS6140_A_MacLeay.utils.GradientDescent as gd
import CS6140_A_MacLeay.utils.NNet as nnet
import hw1
import numpy as np
import pandas as pd

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

def linear_reg(df_train, df_test, Y):
    binary = utils.check_binary(df_train[Y])
    columns = df_train.columns[:-1]
    Y_fit = mystats.linear_regression_points(df_train[columns], df_train[Y])

    #print 'Y_fit'
    #print Y_fit
    #for i in range(0, len(Y_fit)):
    #    print str(Y_fit[i]) + ' -- ' + str(train['is_spam'][i])

    col_MSE = {}
    predict = []

    for i, col in enumerate(columns):
        predict = Y_fit[i] + Y_fit[-1]
    error_train = mystats.get_error(predict, df_train[Y], binary)

    # TEST #
    columns = df_test.columns[:-1]
    Y_fit = mystats.linear_regression_points(df_test[columns], df_test[Y])

    #print 'Y_fit'
    #print Y_fit
    #for i in range(0, len(Y_fit)):
    #    print str(Y_fit[i]) + ' -- ' + str(train['is_spam'][i])

    col_MSE = {}
    predict = []

    for i, col in enumerate(columns):
        predict = Y_fit[i] + Y_fit[-1]
    error_test = mystats.get_error(predict, df_test[Y], binary)
    return [error_train, error_test]

def linear_ridge(df_train, df_test, Y):
    binary = utils.check_binary(df_train[Y])
    columns = df_train.columns[:-1]
    Y_fit = mystats.linear_ridge_points(df_train[columns], df_train[Y])

    #print 'Y_fit'
    #print Y_fit
    #for i in range(0, len(Y_fit)):
    #    print str(Y_fit[i]) + ' -- ' + str(train['is_spam'][i])

    col_MSE = {}
    predict = []

    for i, col in enumerate(columns):
        predict = Y_fit[i] + Y_fit[-1]
    error_train = mystats.get_error(predict, df_train[Y], binary)

    # TEST #
    columns = df_test.columns[:-1]
    Y_fit = mystats.linear_regression_points(df_test[columns], df_test[Y])

    #print 'Y_fit'
    #print Y_fit
    #for i in range(0, len(Y_fit)):
    #    print str(Y_fit[i]) + ' -- ' + str(train['is_spam'][i])

    col_MSE = {}
    predict = []

    for i, col in enumerate(columns):
        predict = Y_fit[i] + Y_fit[-1]
    error_test = mystats.get_error(predict, df_test[Y], binary)
    return [error_train, error_test]

def linear_gd(df_train, df_test, Y):
    """ linear gradient descent """
    binary = utils.check_binary(df_train[Y])
    model = gd.gradient(df_train, df_train[Y], .1, max_iterations=5000)
    print model
    #m, b = gd.get_slope_intercept(model, df_train[Y], binary)
    #print 'm: {} b: {}'.format(m, b)
    #predict = gd.predict(df_train, m, b)
    predict = gd.predict(df_train, model)
    print predict
    error_train = mystats.get_error(predict, df_train[Y], binary)
    predict = gd.predict(df_test, model)
    print predict
    raw_input()
    error_test = mystats.get_error(predict, df_test[Y], binary)
    #error_test = 'none'
    #error_train = 'none'
    return [error_train, error_test]

def logistic_gd(df_train, df_test, Y):
    """ logistic gradient descent """
    binary = utils.check_binary(df_train[Y])
    predict = []
    error_train = mystats.get_error(predict, df_train[Y], binary)
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
    h_results.append(dec_or_reg_tree(h_train, h_test, 'MEDV'))
    h_results.append(linear_reg(h_train, h_test, 'MEDV'))
    h_results.append(linear_ridge(h_train, h_test, 'MEDV'))
    h_results.append(linear_gd(h_train, h_test, 'MEDV'))
    h_results.append(logistic_gd(h_train, h_test, 'MEDV'))

    s_test, s_train = utils.split_test_and_train(utils.load_and_normalize_spam_data())
    s_results.append(dec_or_reg_tree(s_train, s_test, 'is_spam'))
    s_results.append(linear_reg(s_train, s_test, 'is_spam'))
    s_results.append(linear_ridge(s_train, s_test, 'is_spam'))
    s_results.append(linear_gd(s_train, s_test, 'is_spam'))
    s_results.append(logistic_gd(s_train, s_test, 'is_spam'))
    print_results_1(s_results, h_results)


def q_2():
    """ Perceptron """
    test, train = utils.load_perceptron_data()
    print train.head(5)
    train_perceptron(train, 4, .05)

def train_perceptron(data, predict, learning_rate):
    max_iterations = 5
    ct_i = 0
    size = len(data)
    cols = []
    for col in data.columns:
        if col != predict:
            cols.append(col)
    X = data[cols]

    # Add column of ones
    X['ones'] = np.ones(size)
    X = X.reindex()
    p = data[predict]

    # keep track of the mistakes
    last_m = 10000000000000

    # Switch x values from positive to negative if y < 0
    ct_neg_1 = 0
    print p[:5]
    for i, row in enumerate(X.iterrows()):
        if list(p)[i] < 0:
            ct_neg_1 += 1
            for cn, col in enumerate(X.columns):
                X.iloc[i, cn] *= -1

    #print 'ct neg is {} '.format(ct_neg_1)
    #print size

    # Get random array of w values
    w = mystats.init_w(5)[0]
    print 'w array'
    print w.head(5)
    print X.head(5)

    while ct_i < max_iterations:  # for each iteration

        J = []
        n_row = 0
        mistakes = pd.DataFrame(columns=X.columns)
        mistakes_x_sum = 0
        print 'w'
        print w
        for r_ct, row in X.iterrows():  # for each row
            x_sum = 0
            #print 'ct_i {} j {} w {} x {} x_sum {}'.format(ct_i, n_row, wj, X.iloc[n_row][ct_i], x_sum)
            for c_ct, col in enumerate(X.columns):
                x_sum += w[c_ct] * row[col]
            #if n_row < 5:
            #    print x_sum
            n_row += 1
            J.append(x_sum)
            if x_sum < 0:
                mistakes.loc[len(mistakes)] = row
                mistakes_x_sum += x_sum
                #print 'mistakes len {}'.format(len(mistakes))
                #print mistakes.head(5)

        # Check dot product (paranoia)
        #print 'J'
        #print J[:5]
        #print 'dot'
        #print np.dot(X,w)[:5]

        # check objective
        print 'sum of J is {}'.format(sum(J))
        print 'iteration: {} length: {} sum: {}'.format(ct_i, len(mistakes), -1 * mistakes_x_sum)


        print '{} mis*lr={}'.format(mistakes_x_sum, mistakes_x_sum * learning_rate)


        #print w.head(5)

        # update w
        for wi, wcol in enumerate(mistakes.columns):
            # Add the sum of mistakes for each column to w for that column
            w[wi] += learning_rate * sum(mistakes[wcol])
            print 'wcol: {} {}'.format(wcol, sum(mistakes[wcol]))
        #w += sum(mistakes) * learning_rate

        if last_m < (-1 * mistakes_x_sum):
            print 'last_m is {} and size of mistakes is {}'.format(last_m, -1 * mistakes_x_sum)
            break

        last_m = -1 * mistakes_x_sum
        ct_i += 1
        #J = np.dot(w.transpose(), X)
        #w_new = w

    print pd.DataFrame(J).head(5)


def q_3():
    print 'Run Neural Network for question 3 in homework 2'
    nnet.run()


def homework2():
    q_1()
    #q_2()
    #q_3()


if __name__ =='__main__':
    homework2()



