__author__ = 'Allison MacLeay'

import pandas as pd
import numpy as np
import os
import random
# import CS6140_A_MacLeay.utils
# import CS6140_A_MacLeay.utils.Stats as mystats
# import Stats as mystats
import sys


def load_and_normalize_housing_set_preserve_result_col():
    path = '../data/housing'
    test_file = "housing_test.txt"
    train_file = "housing_train.txt"
    test = read_housing_file(os.path.join(path, test_file))
    train = read_housing_file(os.path.join(path, train_file))
    train, test = normalize_train_and_test(train, test, 'MEDV')
    return test, train

def load_and_normalize_housing_set():
    path = '../data/housing'
    test_file = "housing_test.txt"
    train_file = "housing_train.txt"
    test = read_housing_file(os.path.join(path, test_file))
    train = read_housing_file(os.path.join(path, train_file))
    train, test = normalize_train_and_test(train, test, 'MEDV')
    return test, train

def load_perceptron_data():
    path = 'data'
    perceptron_file = 'perceptronData.txt'
    os.system('pwd')
    data = read_perceptron_file(os.path.join(path, perceptron_file))
    return split_test_and_train(data)

def train_subset(df, cols, n=10):
    """" Return a subset of data for debugging """
    sample = random.sample(df.index, n)
    return df.ix[sample][cols].reset_index(drop=True)


def load_and_normalize_spam_data():
    path = os.path.join('../data', 'spambase')
    spamData = read_spam_file(path, 'spambase.data')
    spamData = normalize_data(spamData, 'is_spam')
    return spamData

def split_test_and_train(df, percent=.2, norepeat=True):
    # Alternatively can use the following
    # from sklearn.cross_validation import train_test_split
    # train, test = train_test_split(df, test_size = 0.2)
    # or
    # msk = np.random.rand(len(df)) < 0.8
    number_in_test = len(df) * percent
    test_indeces = np.random.random_integers(len(df)-1, size=(1., number_in_test))[0]
    print 'length of test indeces {} set {}'.format(len(test_indeces), len(set(test_indeces)))

    if norepeat:
        diff = len(test_indeces) - len(set(test_indeces))
        while diff != 0:
            test_indeces = np.append(np.array(list(set(test_indeces))), np.random.random_integers(len(df), size=(1., diff))[0])
            diff = len(test_indeces) - len(set(test_indeces))

    msk = []
    for i in range(0, len(df)):
        if i in test_indeces:
            msk.append(True)
        else:
            msk.append(False)
    msk = np.array(msk)
    test = df[msk]
    train = df[~msk]
    return test, train


def shift_and_scale(df, predict):
    for col in df.axes[1]:
        if col is predict:
            continue
        df[col] = [float(x) for x in df[col]]
        mean = average(df[col])
        df[col] -= mean
        df[col] = df[col]/df[col].max()
        return df


def normalize_data(df, predict):
    # Will not normalize the column defined in 'predict'
    for col in df.axes[1]:
        if col is predict:
            continue
        min = df[col].min()
        df[col] = df[col] - min
        max = df[col].max()
        df[col] = df[col]/max
    return df

def normalize_train_and_test(train, test, skip):
    # Will not normalize the column defined in 'skip'
    #   norm(x[col]) = ( X - min(X) ) / ( max(X) - min(X) )
    tlen = len(train)
    df = pd.concat([train, test])
    for col in df.axes[1]:
        if col is skip:
            continue
        min = df[col].min()
        df[col] = df[col] - min
        max = df[col].max()
        df[col] = df[col]/max
    df_train = df[0:tlen]
    df_test = df[tlen:]
    print 'train length out: ' + str(len(df_train)) + ' = ' + str(len(train))
    print 'test length out: ' + str(len(df_test)) + ' = ' + str(len(test))
    return df_train, df_test

def read_housing_file(f):
    df = read_file(f)
    cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = df.transpose()
    df.columns = cols
    return df

def read_perceptron_file(f):
    df = read_file(f)
    df = df.transpose()
    #print df[0]  # columns
    #print df[:][0]  # rows
    return df


def read_spam_file(path, f):
    df = read_file(os.path.join(path, f), ',')
    cols = read_spam_cols(os.path.join(path, 'spambase.names'))
    df = df.transpose()
    df.columns = cols
    return df


def read_spam_cols(f):
    columns = []
    with open(f, 'r') as fh:
        for line in fh:
            if line.startswith('|'):
                continue
            elif ':' not in line:
                continue
            else:
                columns.append(line.split(':')[0])
    columns.append('is_spam')
    return columns


def read_file(f, delim=None):
    text = {}
    with open(f, 'r') as fh:
        for i, line in enumerate(fh):
            txt = line.strip().split(delim)
            if len(txt) > 0:
                text[i] = txt
    df = pd.DataFrame(text, dtype=np.float)
    return df

def to_col_vec(x):
    return x.reshape((len(x), 1))

def matrix_inverse(matrix):
    return np.linalg.inv(matrix)

def average(arr):
    return float(sum(arr))/len(arr)

def scale(arr, s_min, s_max):
    s_range = s_max - s_min
    a_min = min(arr)
    a_range = max(arr) - a_min
    n_array = []
    for a in arr:
        n_array.append((a - a_min) * s_range/a_range + s_min)
    return n_array

def check_binary(arr):
    is_binary = False
    if len(arr.unique()) < 3:
        is_binary = True
    return is_binary

def sigmoid(n):
    return 1/(1+np.exp(-n))

def variance(arr, d):
    # per column
    # d is the number of rows to calculate
    # the smoothing parameter
    var = np.var(arr)
    var = var + (1/d)  # smoothing
    return var

def add_col(data_rows, n):
    for r in range(len(data_rows)):
        data_rows[r].append(n)
    return data_rows

def get_hw(row, w, binary=False):
    sum = 0
    for jct, j in enumerate(row):
        sum += j * w[jct]
    if binary:
        sum = (1./(1. + np.exp(sum))) -1
    return sum

