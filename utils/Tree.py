__author__ = 'Allison MacLeay'

import numpy as np
import pandas as pd
import utils.Stats as mystats
import utils

def compute_info_gain(df, feature, split, y):
    A = df[[feature, y]]
    #series = [split for x in range(0, len(A[feature]))]
    #print series
    mask = A[feature] <= split
    B = A[mask]
    C = A[~mask]
    info_gain = mystats.binary_entropy(A, y) - mystats.binary_entropy(B, y) + mystats.binary_entropy(C, y)
    #print 'Information Gain: %s' % info_gain
    return info_gain


def compute_info_gain_regression(df, feature, split, y):
    A = df[[feature, y]]
    #series = [split for x in range(0, len(A[feature]))]
    #print series
    mask = A[feature] <= split
    B = A[mask]
    C = A[~mask]
    info_gain = mystats.least_squares(A, y) - mystats.least_squares(B, y) + mystats.least_squares(C, y)
    print 'Information Gain: %s' % info_gain
    return info_gain


def find_best_label_new(df, col, y):
    values = set(df[col])
    print 'vlen: ' + str(len(values))
    mse = {}
    mu = utils.average(df[y])
    for v in values:
        mask = df[col] <= v
        if len(df[mask]) > 0 and len(df[mask]) < len(df):
            mse[v] = mystats.compute_MSE(mu, list(df[mask][col]))
    lkey = min(mse, key=lambda k: mse[k])
    print lkey
    mask = df[col]
    print 'mask len: ' + str(mse[lkey]) +' '+ str(len(mask)) + ' ' + str(len(df))
    return lkey


def find_best_label(df, col, y):
    if col is y:
        return 0
    feature = pd.DataFrame(df[col])
    Y = np.array(df[y])
    y_msk = []
    for i in range(0, len(Y)):
        #print Y[i]
        if Y[i] == 1:
            y_msk.append(True)
        else:
            y_msk.append(False)
    y_msk = np.array(y_msk)
    #print 'mask is '
    #print y_msk
    pos = feature[y_msk]
    #print pos
    #print y_msk
    neg = feature[~y_msk]
    p_mean = pos.mean()
    n_mean = neg.mean()
    h = p_mean[col]
    l = n_mean[col]
    #print 'Positive mean: %s\nNegative mean: %s' % (p_mean, n_mean)
    if l > h:
        h = l
        l = p_mean[col]
    mid = l + float(h - l)/2
    #print 'Midpoint: %s' % str(mid)
    return mid

def find_best_label_regression(df_old, col, y):
    least_sq = {}
    sarray = []
    df = df_old.copy()
    sorted = df.sort([col]).reset_index(drop=True)
    #sorted.reset_index(drop=True)
    #for row in enumerate(df.sort([col])):
    #    sarray.append(row)
    #sorted = pd.DataFrame(sarray)
    #sorted.columns = df.columns
    print 'Print reset index'
    print df[y][0:10]
    print sorted[y][0:10]
    #sys.exit()
    i = 0
    print 'Finding label for ' + col
    for _, row in sorted.iterrows():
        i += 1
        if i == 1 or i > len(sorted) - 1:
            continue
        # print 'i:' + str(i)
        # print list(sorted[y])[i:len(sorted[y])]
        # print 'ls {} + {}'.format(len(list(sorted[y])[0:i]), len(list(sorted[y])[i:len(sorted[y])]))
        lsq = mystats.least_squares(list(sorted[y])[0:i]) + mystats.least_squares(list(sorted[y])[i:len(sorted[y])])
        least_sq[row[col]] = lsq
        # print 'ls {} + {}'.format(str(least_squares(list(sorted[y])[0:i])), least_squares(list(sorted[y])[i:len(sorted[y])]))
    return min(least_sq, key=lambda k: least_sq[k])



def find_best_feature_and_label_for_split(df_full, y, regression):
    # TODO - skip columns with only 1 value
    search_cols = []
    for col in df_full.columns:
        if len(df_full[col].unique()) > 1 and col is not y:
            search_cols.append(col)
    if len(search_cols) < 2:
        return None, None
    df = df_full[search_cols]
    df[y] = df_full[y]
    if len(df) < 3:
        return None, None
    info_gain = pd.DataFrame(np.zeros(len(search_cols)), index=search_cols, columns=['IG'])
    #print info_gain
    labels = {}
    for col in search_cols:
        #if col is y:
        #    continue
        #print 'Col is ' + col
        #print df[col].describe()
        if len(df[col].unique()) < 2:
            #pass
            print 'Warning: column {} has less than 2 unique values'.format(col)
            continue
        if regression:
            #label = find_best_label_regression(df, col, y)
            label = find_best_label_new(df, col, y)
            print ' Label: ' + str(label)
        else:
            #label = find_best_label(df, col, y)
            label = find_best_label_new(df, col, y)
            #label = find_best_label_regression(df, col, y)
        labels[col] = label
        if not regression:
            info_gain[col] = compute_info_gain(df, col, label, y)
        else:
            info_gain[col] = compute_info_gain_regression(df, col, label, y)
    best_col = info_gain['IG'].idxmax()

    print best_col
    print info_gain[best_col]
    if info_gain[best_col] is 0:
        best_col = None

    return best_col, labels[best_col]


class Node:

    def __init__(self, array, level=0, parent=None):
        self.presence = array
        self.population = sum(array)
        self.left = None
        self.right = None
        self.label = {'feature': '', 'criteria': ''}
        self.level = level
        self.parent = parent
        self.predict = None
        self.error = None
        self.test_error = None

    def add_right(self, array):
        self.right = Node(array, self.level + 1, self)

    def add_left(self, array):
        self.left = Node(array, self.level + 1, self)

    def leaf(self, predict, error):
        self.predict = predict
        self.error = error

    def test_leaf(self, error):
        self.test_error = error

    def set_presence(self, array):
        self.presence = array

    def get_node_data(self, df):
        mask = []
        for i in self.presence:
            if i == 1:
                mask.append(True)
            else:
                mask.append(False)
        return df[mask]

    def split(self, feature_name, feature_data, label):
        A = list(self.presence)
        B = list(self.presence)
        self.label = {'feature': feature_name, 'criteria': label}
        for i, f in enumerate(feature_data):
            if self.presence[i] == 1:
                #print str(i) + 'presence is 1 f=' + str(f) + ' label is ' + str(label)
                if f <= label:
                    B[i] = 0
                else:
                    A[i] = 0
        return A, B

    def number_present(self):
        return sum(self.presence)

    def get_print_info(self):
        label_feature = self.label['feature']
        label_criteria = self.label['criteria']
        return 'level: {} feature: {} criteria: {} #: {}'.format(str(self.level), label_feature, label_criteria, str(self.number_present()))

    def show_children_tree(self, follow=True):
        start = self.level
        line = ''
        left = self.left
        right = self.right
        if left is None:
            line = line + 'Left is empty  '
        else:
            line = line + left.get_print_info() + '\t'
        if right is None:
            line = line + 'Right is empty  '
        else:
            line = line + right.get_print_info() + '\t'
        print line
        if follow:
            if left is not None:
                left.show_children_tree()
            if right is not None:
                right.show_children_tree()


class Tree(object):

    def __init__(self, head_node):
        self.leaves = []
        nodes = {}
        self.size = len(head_node.presence)
        self.load_nodes(head_node, nodes)
        print nodes

    def load_nodes(self, node, data):
        key = node.level
        if key not in data.keys():
            data[key] = []
        data[key].append(node)
        if node.predict is not None:
            self.leaves.append(node)
        if node.left is not None:
            self.load_nodes(node.left, data)
        if node.right is not None:
            self.load_nodes(node.right, data)

    def print_leaves(self):
        for leaf in self.leaves:
            print 'predict {} with training error = {}'.format(str(leaf.predict), str(leaf.error))

    def print_leaves_test(self):
        for leaf in self.leaves:
            print 'predict {} with testing error = {}'.format(str(leaf.predict), str(leaf.test_error))


    def error(self):
        total_err = 0
        for leaf in self.leaves:
            total_err += leaf.error
        return float(total_err)/len(self.leaves)

    def error_test(self):
        total_err = 0
        for leaf in self.leaves:
            total_err += leaf.test_error
        return float(total_err)/len(self.leaves)


    def predict_obj(self):
        predict_obj = np.zeros(self.size)
        for leaf in self.leaves:
            predict = leaf.predict
            for i in range(0, len(leaf.presence)):
                if leaf.presence[i] == 1:
                    if predict_obj[i] != 0:
                        print 'ERROR in predict object!'
                    predict_obj[i] = predict

        return predict_obj
