__author__ = 'Allison MacLeay'

import pandas as pd
import numpy as np
import os


def load_and_normalize_housing_set():
    path = '../data/housing'
    test_file = "housing_test.txt"
    train_file = "housing_train.txt"
    test = read_housing_file(os.path.join(path, test_file))
    train = read_housing_file(os.path.join(path, train_file))
    test = normalize_data(test, 'MEDV')  # TODO - normalize test and train together
    train = normalize_data(train, 'MEDV')
    return test, train


def load_and_normalize_spam_data():
    path = os.path.join('../data', 'spambase')
    spamData = read_spam_file(path, 'spambase.data')
    spamData = normalize_data(spamData, 'is_spam')
    return spamData

def split_test_and_train(df, percent=.2):
    # Alternatively can use the following
    # from sklearn.cross_validation import train_test_split
    # train, test = train_test_split(df, test_size = 0.2)
    # or
    # msk = np.random.rand(len(df)) < 0.8
    number_in_test = len(df) * percent
    test_indeces = np.random.random_integers(len(df)-1, size=(1., number_in_test))[0]
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



def normalize_data(df, predict):
    # Will not normalize the column defined in 'predict'
    for col in df.axes[1]:
        if col is predict:
            continue
        min = df[col].min()
        df[col] = df[col] - min
        max = df[col].max()
        df[col] = df[col]/(max-min)
    return df

def read_housing_file(f):
    df = read_file(f)
    cols = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = df.transpose()
    df.columns = cols
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


def matrix_inverse(matrix):
    return np.linalg.inv(matrix)

def average(arr):
    return float(sum(arr))/len(arr)