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

def load_and_normalize_housing_set(skip='MEDV'):
    path = '../data/housing'
    test_file = "housing_test.txt"
    train_file = "housing_train.txt"
    test = read_housing_file(os.path.join(path, test_file))
    train = read_housing_file(os.path.join(path, train_file))
    train, test = normalize_train_and_test(train, test, skip)
    return test, train

def load_perceptron_data():
    path = 'data'
    perceptron_file = 'perceptronData.txt'
    os.system('pwd')
    data = read_perceptron_file(os.path.join(path, perceptron_file))
    return split_test_and_train(data)

def load_gaussian(num):
    path = 'data/hw3'
    filename = str(num) + 'gaussian.txt'
    data = read_gaussian_file(os.path.join(path, filename))
    #print data
    return data


def train_subset(df, cols, n=10):
    """" Return a subset of data for debugging """
    sample = random.sample(df.index, n)
    return df.ix[sample][cols].reset_index(drop=True)


def load_and_normalize_spam_data():
    path = os.path.join('../data', 'spambase')
    spamData = read_spam_file(path, 'spambase.data')
    spamData = remove_constant_column(spamData)
    spamData = normalize_data(spamData, 'is_spam')
    return spamData

def load_and_normalize_polluted_spam_data(set='train'):
    path = os.path.join('data/HW5', 'spam_polluted')
    features = read_file_data(os.path.join(path, set + '_feature.txt')) #, max_rec=300)
    features = remove_constant_column(features)
    features = normalize_data(features, 'is_spam')
    truth = read_file_data(os.path.join(path, set + '_label.txt'))
    return add_row(features, truth)

def load_and_fill_missing_spam_data(dset='train'):
    path = 'data/HW5'
    features = read_file_data(os.path.join(path, '20_percent_missing_' + dset + '.txt')) #, max_rec=300)
    features = correct_missing(features)

    features = remove_constant_column(features)
    features = normalize_data(features, 'is_spam')
    #truth = read_file_data(os.path.join(path, set + '_label.txt'))
    return features

def correct_missing(data, replace=1):
    cor = []
    for i in range(len(data)):
        #row = [float[r] for r in data[i][0].split(',')]
        row = data[i][0].split(',')
        for j in range(len(row)):
            r = float(row[j])
            #if type(r) is str or np.isnan(r):
            #    row[j] = r
            #else:
            #    row[j] = r
        cor.append(row)
    return cor

def remove_constant_column(df):
    varying_columns = []
    if type(df) is list:
        dt = transpose_array(df)
        for i in range(len(dt)):
            if len(set(dt[i])) > 1:
                varying_columns.append(dt[i])
        return transpose_array(varying_columns)
    for col in df.columns:
        if len(df[col].unique()) > 1:
            varying_columns.append(col)
    return df[varying_columns]


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


def normalize_data(df, predict, remove_nan=True):
    # Will not normalize the column defined in 'predict')
    if type(df) is list:
        dt = transpose_array(df)
        for j in range(len(dt)):
            dt[j] = np.asarray(dt[j])
            if remove_nan:
                row, nans, dt[j] = find_and_remove_nans(dt[j])
            else:
                row = dt[j]
            m = np.min(row)
            jmin = [m] * len(row)
            row -= jmin
            m = np.max(row)
            jmax = [m] * len(row)
            row /= jmax
            dt[j] = replace_nans(row, dt[j], dt[-1], nans)
        df = transpose_array(dt)
    else:
        for col in df.axes[1]:
            if col is predict:
                continue
            min = df[col].min()
            df[col] = df[col] - min
            max = df[col].max()
            df[col] = df[col]/max
    return df

def replace_nans(data, polluted, y, nans):
    if len(nans) == 0:
        return data
    polluted = np.asarray(polluted, dtype=float)
    y_1 = [x for x in polluted if x==1]
    y_0 = [x for x in polluted if x!=1 and not np.isnan(x)]
    py_1 = np.mean(y_1)
    py_0 = np.mean(y_0)
    for i in range(len(polluted)):
        if np.isnan(polluted[i]):
            polluted[i] = py_1 if y[i] == 1 else py_0
    return polluted



def find_and_remove_nans(data):
    nan = []
    cleaned = []
    for i, x in enumerate(data):
        try:
            x = float(x)
            data[i] = x
        except Exception as e:
            print e
        if np.isnan(x):
            nan.append(i)
        else:
            cleaned.append(x)
    return np.asarray(cleaned), nan, np.asarray(data)

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

def read_gaussian_file(f):
    df = read_file(f, ' ')
    df = df.transpose()
    #print df[0]  # columns
    #print df[:][0]  # rows
    print 'read_gaussian'
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

def read_file_data(f, delim=None, max_rec=None):
    text = []
    with open(f, 'r') as fh:
        for i, line in enumerate(fh):
            text.append([])
            txt = line.strip().split(delim)
            if len(txt) > 0:
                text[i] = check_type(txt)
            if max_rec is not None and i > max_rec:
                break
    return text

def add_row(X, y, remove_nans=True):
    data = []
    for i in range(len(X)):
        row = X[i]
        if type(row) is np.ndarray:
            row = list(row)
        y_val = y[i] if type(y[i]) is float else y[i][0]
        row.append(y_val)
        if remove_nans and not any(np.isnan(row)):
            data.append(row)
    return data


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

def split_truth_from_data(data):
    """ Assumes that the truth column is the last column """
    truth_rows = transpose_array(data)[-1]  # truth is by row
    data_rows = transpose_array(transpose_array(data)[:-1])  # data is by column
    return truth_rows, data_rows

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

def random_sample(array, y_arr=None, size=100):
    """ return indeces """
    idx = np.random.choice(range(len(array)), size=size)
    data = [array[i] for i in idx]
    if y_arr is not None:
        y = [y_arr[i] for i in idx]
    else:
        y = None
    return data, y

def check_type(v):
    for i in range(len(v)):
        try:
            v[i] = float(v[i])
        except Exception as e:
            pass
            #print 'Data is type {} not float! (at {})'.format(type(v[i]), i)
    return v

