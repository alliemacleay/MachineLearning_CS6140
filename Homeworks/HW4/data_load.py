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

def metadata_q4():
    folder = 'data/8newsgroup'
    path = os.path.join(folder, 'train.trec')
    data = read_file(os.path.join(path, 'feature_settings.txt'), ',')
    data = get_feature_settings(data)
    return data

def metadata_q4_labels():
    folder = 'data/8newsgroup'
    path = os.path.join(folder, 'train.trec')
    data = read_file(os.path.join(path, 'data_settings.txt'), ',')
    data = get_label_settings(data)
    return data

def get_label_settings(data):
    dsettings = {}
    for i in range(len(data)):
        for j in range(len(data[i])):
            srow = data[i][j].split('=')
            if len(srow) != 2:
                print srow
            else:
                ftype = srow[0]
                fval = srow[1]
            if ftype == 'intId':
                idx = int(fval)
            elif ftype == 'extLabel':
                name = fval
        if name not in dsettings.keys():
            dsettings[name] = [idx]
        else:
            dsettings[name].append(idx)

    return dsettings

def get_feature_settings(data):
    fsettings = {}
    for i in range(len(data)):
        for j in range(len(data[i])):
            srow = data[i][j].split('=')
            if len(srow) != 2:
                print srow
            else:
                ftype = srow[0]
                fval = srow[1]
            if ftype == 'featureIndex':
                idx = int(fval)
            elif ftype == 'featureName':
                name = fval
        if name not in fsettings.keys():
            fsettings[name] = [idx]
        else:
            fsettings[name].append(idx)
    return fsettings


def data_q4():
    folder = 'data/8newsgroup'
    path = os.path.join(folder, 'train.trec')
    data = read_file(os.path.join(path, 'feature_matrix.txt'), ' ')
    data, feature = feature_map(data)
    #data = clean_data(data)
    #data = normalize_data(data)
    return data, feature

def get_data_with_ft(data, fmap, feature_list):
    idx = fmap[feature_list[0]]
    sub = []
    for i in range(1, len(feature_list)):
        ft = feature_list[i]
            #for id in idx:
            #if ft not in fmap[m]:
            #    has_all = False
        if has_all:
            idx.append(m)
    for i in idx:
        sub.append(data[i])
    return sub, idx

def feature_map(data):
    values = []
    features = []
    for i in range(len(data)):
        ft_row = []
        val_row = []
        for j in range(1, len(data[i])):
            ft, val = data[i][j].split(':')
            ft_row.append(ft)
            val_row.append(val)
        values.append(val_row)
        features.append(ft_row)

    return values, features

def index_features(data):
    features = {}
    for i in range(len(data)):
        for ft in data[i]:
            if ft not in features.keys():
                features[ft] = [i]
            else:
                features[ft].append(i)
    return features


def read_file(infile, delim='\t'):
    X = []
    with open(infile, 'r') as fh:
        for line in fh:
            row = line.strip().split(delim)
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

def random_sample(array, y_arr, size):
    """ return indeces """
    idx = np.random.choice(range(len(array)), size=size)
    data = [array[i] for i in idx]
    y = [y_arr[i] for i in idx]
    return data, y

def load_spirals():
    file = '../data/twoSpirals.txt'
    data = []
    y = []
    with open(file, 'rb') as spirals:
        for line in spirals:
            line = line.strip()
            data.append(np.array(line.split('\t')[:-1], dtype=float))
            y.append(line.split('\t')[-1])
    return np.array(data), np.array(y, dtype=float)





