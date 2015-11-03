__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.utils as utils
import numpy as np
import os

uci_folder = 'data/UCI'

def data_q2():
    pass

def data_q3_crx():
    path = os.path.join(uci_folder, 'crx')
    data = read_file(os.path.join(path, 'crx.data'))
    data = clean_data(data)
    data = normalize_data(data)
    return data

def data_q3_vote():
    path = os.path.join(uci_folder, 'vote')
    data = read_file(os.path.join(path, 'vote.data'))
    data = clean_data(data)
    data = normalize_data(data)
    return data


def data_q4():
    folder = 'data/8newsgroup'
    path = os.path.join(folder, 'train.trec')
    data = read_file(os.path.join(path, 'feature_matrix.txt'))
    data = clean_data(data)
    data = normalize_data(data)
    return data

def read_file(infile):
    X = []
    with open(infile, 'r') as fh:
        for line in fh:
            row = line.strip().split('\t')
            X.append(row)
    return X


def normalize_data(X, skip=None):
    if skip is not None and skip < 0:
        skip += len(X[0])
    by_col = utils.transpose_array(X)
    normalized = []
    for j in range(len(by_col)):
        if skip != j:
            new_col, is_singular = normalize_col(by_col[j])
            normalized.append(new_col)
    return utils.transpose_array(normalized)

def clean_data(X, remove_constant=False):
    by_col = utils.transpose_array(X)
    nan_rows = []
    new_by_col = []
    for i, x in enumerate(by_col):
        col, bad_rows, is_singular = check_type(x)
        for b in bad_rows:
            if b not in nan_rows:
                nan_rows.append(b)
        if not is_singular or not remove_constant:
            new_by_col.append(col)
    upright = utils.transpose_array(new_by_col)
    new_X = []
    for i, row in enumerate(upright):
        if i not in nan_rows:
            new_X.append(row)
    return new_X

def normalize_col(col):
    is_singular = False
    cmin = min(col)
    cmax = max(col)
    if cmin == cmax:
        is_singular = True
    col = [i - cmin for i in col]
    cmax = max(col)
    col = [float(i)/cmax for i in col]
    return col, is_singular


def check_type(col):
    new_col = []
    serialized = False
    classes = {}
    contains_nans = []
    is_singular = False
    for i, x in enumerate(col):
        #print x
        if serialized:
            val = serialize(x, classes)
            if val is np.nan:
                contains_nans.append(i)
            new_col.append(val)
        else:
            try:
                new_col.append(float(x))
            except ValueError as e:
                if i == 0:
                    val = serialize(x, classes)
                    if val is np.nan:
                        contains_nans.append(i)
                    new_col.append(val)
                    serialized = True
                else:
                    new_col.append(np.nan)
                    contains_nans.append(i)
    if min(new_col) == max(new_col):
        is_singular = True
    return new_col, contains_nans, is_singular


def serialize(x, classes):
    if x == '?':
        val = np.nan
    elif x in classes.keys():
        val = classes[x]
    else:
        if len(classes.values()) > 0:
            val = len(classes.values())

        else:
            val = 0
        classes[x] = val
    return val

def get_train_and_test(folds, k):
    train = []
    for i, fold in enumerate(folds):
        if i != k:
            for row in folds[i]:
                train.append(row)
    return train, folds[k]



