import os
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import scipy as sp
import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.utils.Stats as mystats
import CS6140_A_MacLeay.utils.plots as myplot
import CS6140_A_MacLeay.utils.Tree as mytree
import sys



def train_regression_tree(df):
    X = df
    Y = df['MEDV']
    clf = tree.DecisionTreeRegressor()
    clf.fit(X, Y)
    print 'Result of training regression tree:'
    print clf
    return clf


def test_regression_tree(clf, df):
    result = clf.predict(df)
    print 'Result of test regression tree (hw1 problem 1 dataset 1) :'
    print result
    return result

def train_decision_tree(df):
    # TODO
    X = df
    Y = df['is_spam']
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    print 'Result of training decision tree'
    print clf
    return clf

def test_decision_tree(clf, df):
    result = clf.predict(df)
    print 'Result of test decision tree (hw1 problem 1 dataset 2) :'
    print result
    return result


def regression_housing_set():
    """
    Solution for HW1 prob 1
    """
    print('Homework 1 problem 1 - Regression Decision tree')
    print('Housing Dataset')
    test, train = utils.load_and_normalize_housing_set()
    dt_reg = train_regression_tree(train)
    predicted = test_regression_tree(dt_reg, test)
    error = mystats.calculate_chisq_error(predicted, test['MEDV'])
    print 'Error: ' + str(error)


def decision_spambase_set():
    """
    Solution for HW1 prob 1
    """
    print('Homework 1 problem 1 - Regression Decision tree')
    print('Spambase Dataset')
    spam_data = utils.load_and_normalize_spam_data()
    test, train = utils.split_test_and_train(spam_data)
    print str(len(train)) + " # in training set <--> # in test " + str(len(test))
    dt = train_decision_tree(train)
    predicted = test_decision_tree(dt, test)
    #print predicted
    #print test['is_spam']
    error = mystats.calculate_binary_error(predicted, test['is_spam'])
    print 'Error: ' + str(error)

def decision_spambase_set_no_libs():
    """
    Solution for HW1 prob 1
    """
    print('Homework 1 problem 1 - No Libraries - Regression Decision tree')
    print('Spambase Dataset')
    spam_data = utils.load_and_normalize_spam_data()
    test, train = utils.split_test_and_train(spam_data)
    print str(len(train)) + " # in training set <--> # in test " + str(len(test))

    node = mytree.Node(np.ones(len(train)))
    branch_node(node, train, 5, 'is_spam')
    #node.show_children_tree()
    node.show_children_tree(follow=False)

    model = mytree.Tree(node)
    model.print_leaves()
    print 'Trained model error is : ' + str(model.error())

    node.presence = np.ones(len(test))
    test_node(node, test, 'is_spam')
    test_tree = mytree.Tree(node)
    prediction = test_tree.predict_obj()
    test_tree.print_leaves_test()
    print 'predict sum: ' + str(sum(prediction))
    print 'MSE:' + str(test_tree.error_test())

    [tp, tn, fp, fn] = mystats.get_performance_stats(test['is_spam'].as_matrix(), prediction)
    print 'TP: {}\tFP: {}\nTN: {}\tFN: {}'.format(tp, fp, tn, fn)
    print 'Accuracy: ' + str(mystats.compute_accuracy(tp,tn, fp,fn))
    print 'MSE: ' + str(mystats.compute_MSE_arrays(prediction, test['is_spam']))

def decision_housing_set_no_libs():
    """
    Solution for HW1 prob 1
    """
    print('Homework 1 problem 1 - No Libraries - Regression Decision tree')
    print('Housing Dataset')
    test, train = utils.load_and_normalize_housing_set()

    # The following 2 lines are for debugging
    #train = utils.train_subset(train, ['ZN','CRIM', 'TAX', 'DIS', 'MEDV'], n=50)
    #test = utils.train_subset(test, ['ZN', 'CRIM', 'TAX', 'DIS', 'MEDV'], n=3)

    print str(len(train)) + " # in training set <--> # in test " + str(len(test))
    node = mytree.Node(np.ones(len(train)))
    branch_node(node, train, 2, 'MEDV', regression=True)
    #node.show_children_tree()
    node.show_children_tree(follow=False)

    model = mytree.Tree(node)
    model.print_leaves()
    model.print_tree(train)
    print 'Trained model error is : ' + str(model.error())
    train_prediction = model.predict_obj()
    print 'Training MSE is: ' + str(mystats.compute_MSE_arrays(train_prediction, train['MEDV']))
    sys.exit()

    node.presence = np.ones(len(test))
    test_node(node, test, 'MEDV', regression=True)
    test_tree = mytree.Tree(node)
    prediction = test_tree.predict_obj()
    #raw_input()
    print 'predict sum: ' + str(sum(prediction))
    test_tree.print_leaves_test()
    print 'ERROR: ' + str(test_tree.error_test())
    print prediction
    print 'train'
    print train['MEDV']
    print 'test'
    print test['MEDV']
    MSE = mystats.compute_MSE_arrays(prediction, test['MEDV'])
    print 'MSE: ' + str(MSE)
    print 'RMSE: ' + str(np.sqrt(MSE))

    test_tree.print_tree(test, long=False)

    #[tp, tn, fp, fn] = mystats.get_performance_stats(test['MEDV'].as_matrix(), prediction)
    #print 'TP: {}\tFP: {}\nTN: {}\tFN: {}'.format(tp, fp, tn, fn)


def regression_line_spam_no_libs():
    """
    Solution for HW1 prob 2
    """
    print('Homework 1 problem 2 - No Libraries - Regression Line')
    print('Spam Dataset')
    spam_data = utils.load_and_normalize_spam_data()
    test, train = utils.split_test_and_train(spam_data)
    columns = train.columns[:-1]
    Y_fit = mystats.linear_regression_points(train[columns], train['is_spam'])

    #print 'Y_fit'
    #print Y_fit
    #for i in range(0, len(Y_fit)):
    #    print str(Y_fit[i]) + ' -- ' + str(train['is_spam'][i])

    col_MSE = {}
    for i, col in enumerate(columns):
        col_fit = Y_fit[i] + Y_fit[-1]
        col_MSE[col] = mystats.compute_MSE_arrays(col_fit, train['is_spam'])
    print col_MSE
    RMSE = np.sqrt(col_MSE.values())
    average_MSE = utils.average(col_MSE.values())
    average_RMSE = utils.average(RMSE)
    print 'Average MSE: ' + str(average_MSE)
    print 'Average RMSE: ' + str(average_RMSE)


def regression_line_housing_no_libs():
    """
    Solution for HW1 prob 2
    """
    print('Homework 1 problem 2 - No Libraries - Regression Line')
    print('Housing Dataset')
    test, train = utils.load_and_normalize_housing_set()
    print str(len(train)) + " # in training set <--> # in test " + str(len(test))
    columns = train.columns[:-1]
    Y_fit = mystats.linear_regression_points(train[columns], train['MEDV'])
    print 'Y_fit'
    print Y_fit
    #for i in range(0, len(Y_fit)):
    #    print str(Y_fit[i]) + ' -- ' + str(train['MEDV'][i])

    row_sums = np.zeros(len(Y_fit[0]))
    for col in Y_fit:
        for i in range(0, len(col)):
            row_sums[i] += col[i]

    print row_sums

    col_MSE = {}
    for i, col in enumerate(columns):
        col_fit = row_sums[i]  # Y_fit[i] + Y_fit[-1]
        col_MSE[col] = mystats.compute_MSE(col_fit, train['MEDV'])
    print col_MSE
    RMSE = np.sqrt(col_MSE.values())
    average_MSE = utils.average(col_MSE.values())
    average_RMSE = utils.average(RMSE)
    print 'Average MSE: ' + str(average_MSE)
    print 'Average RMSE: ' + str(average_RMSE)



def test_regression_line_housing_no_libs():
    """
    Testing 2 variable solution for HW1 prob 2
    """
    print('Testing linear regression with 2 columns')
    test, train = utils.load_and_normalize_housing_set()
    print str(len(train)) + " # in training set <--> # in test " + str(len(test))
    columns = train.columns[:-1]
    Y_fit = mystats.linear_regression_points(train[columns[0]], train['MEDV'])
    #for i, col in enumerate(columns):
    print 'Y_fit'
    print Y_fit
    for i in range(0, len(Y_fit)):
        print str(Y_fit[i]) + ' -- ' + str(train['MEDV'][i])
    print train[columns[0]]
    #myplot.points([train[columns[0]], train['MEDV']])

    #myplot.points([train[columns[0]], list(Y_fit[0])])
    myplot.fit_v_point([train[columns[0]], train['MEDV'], list(Y_fit[0] + Y_fit[-1])])
    col_MSE = {}
    print columns[0]
    i = 0
    col = 'CRIM'
    col_fit = Y_fit[i] + Y_fit[-1]
    col_MSE[col] = mystats.compute_MSE_arrays(col_fit, train['MEDV'])
    print col_MSE






def branch_node(node, df, threshold, Y, regression=False):
    """
    :param node: Node object defined in Stats
    :param df: The dataframe being used by the tree
    :param threshold: max branching depth
    :param Y: Feature to predict
    :return: void
    """
    print 'Branching Level : ' + str(node.level)
    data = node.get_node_data(df)
    print 'Length of data ' + str(len(data)) + ' len df: ' + str(len(df))
    feature, label = mytree.find_best_feature_and_label_for_split(data, Y, regression)
    print 'feature: {} label: {}'.format(feature, label)
    if feature is not None and node.level < threshold:
        A_array, B_array = node.split(feature, df[feature], label)
        print ' A : {} B: {}'.format(sum(A_array), sum(B_array))
        node.add_left(A_array)
        node.add_right(B_array)
        branch_node(node.left, df, threshold, Y, regression)
        branch_node(node.right, df, threshold, Y, regression)
    else:
        if not regression:
            predict = 0
            prob = mystats.binary_probability(data, Y)
            print 'PROBABILITY ' + str(prob)
            if prob >= .5:
                predict = 1
            error = mystats.binary_error(data, Y, predict)
        else:
            print str(feature) +'is fueaturea ' + str(label) + str(node.presence)
            predict = sum(data[Y])/len(data[Y])
            error = mystats.compute_MSE(predict, list(data[Y]))
        node.leaf(predict, error)

def test_node(node, df, Y, regression=False):
    """
    :param node: Node object defined in Stats
    :param df: The dataframe being used by the tree
    :param Y: Feature to predict
    :return: void
    """
    print 'Testing Branching Level : ' + str(node.level)
    data = node.get_node_data(df)
    print 'Length of TEST data ' + str(len(data)) + ' len df: ' + str(len(df))
    feature = node.label['feature']
    label = node.label['criteria']
    if feature is not '':
        print 'feature ' + feature
        #print df[feature]
        A_array, B_array = node.split(feature, df[feature], label)
        print 'Test A : {} B: {}'.format(sum(A_array), sum(B_array))
        node.left.set_presence(A_array)
        node.right.set_presence(B_array)
        if node.left is not None:
            test_node(node.left, df, Y, regression)
        if node.right is not None:
            test_node(node.right, df, Y, regression)
    else:
        predict = node.predict
        if not regression:
            error = mystats.binary_error(data, Y, predict)
        else:
            error = mystats.compute_MSE(predict, list(data[Y]))
        node.test_leaf(error)


def analyze_spambase_hw1():
    """ HW1 - problem 2 """
    spamData = utils.load_and_normalize_spam_data()
    mystats.k_folds(spamData, 10)


def homework1():
    # -- Uses sklearn to analyze datasets --
    #regression_housing_set()
    #decision_spambase_set()
    #analyze_spambase_hw1()  # kfolds from last year's homework

    # -- Maual implementation of ML algorithms --

    # -- Decision Trees --
    #decision_spambase_set_no_libs()
    decision_housing_set_no_libs()

    # -- Test 2D regression line and plot --
    #test_regression_line_housing_no_libs()

    # -- Linear Regression --
    #regression_line_housing_no_libs()
    #regression_line_spam_no_libs()

