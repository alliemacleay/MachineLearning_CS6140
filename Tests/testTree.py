__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.utils.Tree as tree
import CS6140_A_MacLeay.utils.GradientDescent as gd
import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.utils.plots as plot
import CS6140_A_MacLeay.Homeworks.hw2_new as hw2
import CS6140_A_MacLeay.utils.Stats as mystats
import numpy as np
import pandas as pd
import random
import sys
from nose.tools import assert_equal, assert_true, assert_not_equal, assert_false

def testTree():
    best = ('A', 5)
    data = {'A': [1,2,6,7,8,9,3,4,5], 'C': [1,0,1,0,1,0,1,0,1], 'B': [1,1,0,0,0,0,1,1,1]}
    df = pd.DataFrame(data)
    print tree.find_best_label_new(df, 'A', 'B')
    print 'best feature and label'
    print tree.find_best_feature_and_label_for_split(df, 'B', regression=True)
    #assert_equal(best, tree.find_best_feature_and_label_for_split(df, 'B', regression=True))

def testGradient():  # Great success with subset
    test, train = utils.load_and_normalize_housing_set()
    df_full = pd.DataFrame(train)
    subset_size = 100
    df = utils.train_subset(df_full, ['CRIM', 'TAX', 'B', 'MEDV'], n=subset_size)
    dfX = pd.DataFrame([df['CRIM'], df['TAX']]).transpose()
    print len(dfX)
    print dfX
    #raw_input()

    fit = gd.gradient(dfX, df['MEDV'].head(subset_size), .5, max_iterations=300)

    print 'read v fit'
    print len(dfX)
    print df['MEDV'].head(10)
    print fit
    data = gd.add_col(gd.pandas_to_data(dfX), 1)
    print np.dot(data, fit)

def testGradSynth():
    data, y = get_test_data()
    df = pd.DataFrame(data, columns=["x0", "x1"])
    print gd.gradient(df, y, .5, max_iterations=30)
    pass

def testGradientByColumn():
    test, train = utils.load_and_normalize_housing_set()
    blacklist = ['NOX', 'RM']
    df_full = pd.DataFrame(train)
    for i in range(2, len(df_full.columns) - 1):
        cols = []
        for j in range(1, i):
            if df_full.columns[j] not in blacklist:
                cols.append(df_full.columns[j])
        cols.append('MEDV')
        print cols
        raw_input()
        testGradient_by_columns(df_full, cols)

def testGradient_by_columns(df, cols):  # fail
    df = utils.train_subset(df, cols, n=len(df))
    #dfX = pd.DataFrame([df['CRIM'], df['TAX']]).transpose()
    print len(df)
    print df
    #raw_input()

    fit = gd.gradient(df, df['MEDV'].head(len(df)), .00001, max_iterations=5000)
    print 'read v fit'
    print len(df)
    print df['MEDV'].head(10)
    print fit
    print np.dot(df, fit)

def testGradient2():
    X = np.random.random(size=[10, 2])
    y = .5 * X[:, 0] + 2 * X[:, 1] + 3
    df = pd.DataFrame(data=X)
    w = gd.gradient(df, y, .05)

def testHW2_subset(): # Success
    test, train = utils.load_and_normalize_housing_set()
    df_full = pd.DataFrame(train)
    df_test = utils.train_subset(df_full, ['CRIM', 'TAX', 'B', 'MEDV'], n=10)
    df_train = utils.train_subset(df_full, ['CRIM', 'TAX', 'B', 'MEDV'], n=10)
    dfX_test = pd.DataFrame([df_test['CRIM'], df_test['TAX'], df_test['MEDV']]).transpose()
    dfX_train = pd.DataFrame([df_train['CRIM'], df_train['TAX'], df_train['MEDV']]).transpose()
    print hw2.linear_gd(dfX_train, dfX_test, 'MEDV')

def testHW2_allcols():  # Fail
    test, train = utils.load_and_normalize_housing_set()
    df_full = pd.DataFrame(train)
    cols = [col for col in df_full.columns if col != 'MEDV']
    df_test = utils.train_subset(df_full, cols, n=10)
    df_train = utils.train_subset(df_full, cols, n=10)
    #dfX_test = pd.DataFrame([df_test['CRIM'], df_test['TAX'], df_test['MEDV']]).transpose()
    #dfX_train = pd.DataFrame([df_train['CRIM'], df_train['TAX'], df_train['MEDV']]).transpose()
    print hw2.linear_gd(df_train, df_test, 'MEDV')

def testHW2():  # Success
    test, train = utils.load_and_normalize_housing_set()
    df_train = pd.DataFrame(train)
    df_test = pd.DataFrame(test)
    print df_train.head(10)
    #raw_input()
    print hw2.linear_gd(df_train, df_test, 'MEDV')

def testLogisticGradient():
    """ logistic gradient descent """
    df_test, df_train = utils.split_test_and_train(utils.load_and_normalize_spam_data())
    Y = 'is_spam'
    binary = utils.check_binary(df_train[Y])
    model = gd.logistic_gradient(df_train, df_train[Y], .1, max_iterations=5)
    #print model
    #raw_input()
    predict = gd.predict(df_train, model, binary, True)
    print predict
    error_train = mystats.get_error(predict, df_train[Y], binary)
    #raw_input()
    predict = gd.predict(df_test, model, binary, True)
    print predict
    error_test = mystats.get_error(predict, df_test[Y], binary)
    print 'error train {} error_test {}'.format(error_train, error_test)
    return [error_train, error_test]



def testScale():
    test, train = utils.load_and_normalize_housing_set()
    df_full = pd.DataFrame(train)
    df = utils.train_subset(df_full, ['CRIM', 'TAX', 'B', 'MEDV'], n=10)
    w = []
    for i in range(0,len(df['TAX'])):
        w.append(random.random())
    scaled = utils.scale(w, min(df['TAX']), max(df['TAX']))
    plot.fit_v_point([w, df['MEDV'], scaled])

def testLinRidge_test_data():
    dX, y = get_test_data()
    X = pd.DataFrame(data=dX, columns=["x0", "x1"])
    X['y'] = y
    #print hw2.linear_reg_errors(h_train, h_test, 'MEDV', True)
    print hw2.linear_reg(X, 'y', False, True)

def testLinRidge():
    h_test, h_train = utils.load_and_normalize_housing_set()
    #print hw2.linear_reg_errors(h_train, h_test, 'MEDV', True)
    print hw2.linear_reg(h_train, 'MEDV', False, False)

def testBinary():
    not_binary = [5,6,7]
    binary = [1,0,1]
    df = pd.DataFrame({'not_binary': not_binary, 'binary': binary})
    print df['not_binary']
    nb_result = utils.check_binary(df['not_binary'])
    b_result = utils.check_binary(df['binary'])
    print 'not binary: {} binary: {}'.format(nb_result, b_result)

def testLogGradient2():
    X = np.random.random(size=[10, 2])
    y = utils.sigmoid(X[:, 0]* .5 + 2 * X[:, 1] + 3)
    df = pd.DataFrame(data=X)
    w = gd.logistic_gradient(df, y, .05)
    print w

def UnitTest():
    test_vector_equality()
    test_column_product()
    test_dot_prod()

def test_dot_prod():
    data = {1:[1, 2, 3, 4, 5],
            2:[1, 2, 3, 4, 5],
            3:[1, 2, 3, 4, 5],
            4:[1, 2, 3, 4, 5]}
    multiplier = [2, 2, 2, 2, 2]
    truth = [30, 30, 30, 30]
    assert_true(mystats.check_vector_equality(truth, mystats.dot_product_sanity(data, multiplier)))

def test_vector_equality():
    v1 = [1, 2, 3, 4, 5]
    v2 = [1, 2, 3, 4, 5]
    v3 = [1, 2, 3, 4, 6]
    v4 = [1, 2, 3, 4]
    assert_true(mystats.check_vector_equality(v1, v2))
    assert_false(mystats.check_vector_equality(v2, v3))
    assert_false(mystats.check_vector_equality(v3, v4))

def test_column_product():
    v1 = [1, 2, 3, 4, 5]
    v2 = [2, 2, 2, 2, 2]
    truth = 30
    prod = mystats.column_product(v1, v2)
    assert_equal(truth, prod)

def test_add_col():
    array = []
    for i in range(5):
        row = []
        for j in range(3):
            row.append(0)
        array.append(row)
    print array
    array = utils.add_col(array, 1)
    print array

def get_test_data():
    X = np.random.random(size=(10, 2))
    y = .5 * X[:, 0] + 2 * X[:, 1] + 3
    return X, y

if __name__ == '__main__':
    print 'Test main for HW2'
    #testTree()
    #testScale()
    #test_add_col()
    #testGradient()
    testGradSynth()
    #testGradientByColumn()
    #testGradient2()
    #testLinRidge()
    #testLinRidge_test_data()
    #testLogisticGradient()
    #testLogGradient2()
    #testHW2_allcols()
    #testHW2()
    #testBinary()
    #UnitTest()
