import os
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import scipy as sp
import utils
import utils.Stats as mystats



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
    node = mystats.Node(np.ones(len(train)))
    branch_node(node, train, 8, 'is_spam')
    #node.show_children_tree()
    node.show_children_tree(follow=False)

    model = mystats.Tree(node)
    model.print_leaves()
    print 'Trained model error is : ' + str(model.error())

    node.presence = np.ones(len(test))
    test_node(node, test, 'is_spam')
    test_tree = mystats.Tree(node)
    prediction = test_tree.predict_obj()
    print 'predict sum: ' + str(sum(prediction))
    print test_tree.error()

    [tp, tn, fp, fn] = mystats.get_performance_stats(test['is_spam'].as_matrix(), prediction)
    print 'TP: {}\tFP: {}\nTN: {}\tFN: {}'.format(tp, fp, tn, fn)


def branch_node(node, df, threshold, Y):
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
    feature, label = mystats.find_best_feature_and_label_for_split(data, Y)
    if feature is not None and node.level < threshold:
        A_array, B_array = node.split(feature, df[feature], label)
        print ' A : {} B: {}'.format(sum(A_array), sum(B_array))
        node.add_left(A_array)
        node.add_right(B_array)
        branch_node(node.left, df, threshold, Y)
        branch_node(node.right, df, threshold, Y)
    else:
        predict = 0
        prob = mystats.binary_probability(data, Y)
        print 'PROBABILITY ' + str(prob)
        if prob >= .5:
            predict = 1
        error = mystats.binary_error(data, Y, predict)
        node.leaf(predict, error)

def test_node(node, df, Y):
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
            test_node(node.left, df, Y)
        if node.right is not None:
            test_node(node.right, df, Y)
    else:
        predict = node.predict
        error = mystats.binary_error(data, Y, predict)
        node.test_leaf(error)




def k_folds(df, k):
    """ k folds for hw1 prob 2"""
    number = np.floor(len(df)/k)
    print number
    folds = []
    #for i in range(0, k):
    #    folds.append(df.sample(number, replace=False))
    kf = KFold(len(df), n_folds=k)
    for train, test in kf:
        print test
        print train
    return folds


def analyze_spambase_hw1():
    """ HW1 - problem 2 """
    spamData = utils.load_and_normalize_spam_data()
    k_folds(spamData, 10)


def homework1():
    #regression_housing_set()
    #decision_spambase_set()
    decision_spambase_set_no_libs()
    #analyze_spambase_hw1()
