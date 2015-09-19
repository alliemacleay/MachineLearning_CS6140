__author__ = 'Allison MacLeay'
import numpy as np
import pandas as pd


def compute_ACC(predicted, observed):
    #TO DO - fix this implementation
    chi_sq_errors = calculate_chisq_error(predicted, observed)
    return chi_sq_errors

def compute_MSE(predected, observed):
    #TO DO - fix this implementation
    chi_sq_errors = calculate_chisq_error(predicted, observed)
    return chi_sq_errors

def calculate_chisq_error(pred, truth):
    """ (E-O)^2/E """
    i = 0
    err = 0
    for p in pred:
        t = truth[i]
        err += (t - p)**2/t
        i += 1
    return err/len(truth)

def calculate_binary_error(pred, truth):
    total = len(pred)
    positves_predicted = sum(pred)
    true_positive = sum(np.logical_and(pred, truth))
    true_negative = sum(np.logical_and(np.logical_not(pred), np.logical_not(truth)))
    correct = true_negative + true_positive
    error = float(total - correct)/total
    print 'Total: %s' % total
    print 'True Positive: %s' % true_positive
    print 'True Negative: %s' % true_negative
    print 'Positives Predicted: %s' % positves_predicted
    print 'Correctly Predicted: %s' % correct
    print 'Error: %s' % error
    return error


def binary_info_gain(df, feature, y):
    """
    :param df: input dataframe
    :param feature: column to investigate
    :param y: column to predict
    :return: information gain from binary feature column
    """
    return sum(np.logical_and(df[feature], df[y]))/len(df[feature])

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


def get_performance_stats(truth, predict):
    print 'len: ' + str(len(truth)) + ' : ' + str(len(predict))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(truth)):
        if predict[i] == 1:
            if truth[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if truth[i] == 0:
                tn += 1
            else:
                fn += 1
    return [tp, tn, fp, fn]




def binary_probability(df, y):
    """ probability for features with 0 and 1 values """
    if len(df[y]) is 0:
        prob = 0
    else:
        prob = sum(df[y])/len(df[y])
    return prob

def binary_entropy(df, y):
    """ entropy for features with 0 and 1 values """
    return binary_probability(df, y) * len(df)


def binary_error(df, y, predicted):
    error = binary_probability(df, y)
    if predicted is 1:
        error = 1 - error
    return error


def compute_info_gain(df, feature, split, y):
    A = df[[feature, y]]
    #series = [split for x in range(0, len(A[feature]))]
    #print series
    mask = A[feature] < split
    B = A[mask]
    C = A[~mask]
    info_gain = binary_entropy(A, y) - binary_entropy(B, y) + binary_entropy(C, y)
    #print 'Information Gain: %s' % info_gain
    return info_gain


def find_best_feature_and_label_for_split(df, y):
    if len(df) is 0:
        return None, None
    info_gain = pd.DataFrame(np.zeros(len(df.columns)), index=df.columns, columns=['IG'])
    #print info_gain
    labels = {}
    for col in df.columns:
        if col is y:
            continue
        #print 'Col is ' + col
        #print df[col].describe()
        #print len(df[col].unique())
        label = find_best_label(df, col, y)
        labels[col] = label
        info_gain[col] = compute_info_gain(df, col, label, y)
    best_col = info_gain['IG'].idxmax()

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
                if f < label:
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

    def error(self):
        total_err = 0
        for leaf in self.leaves:
            total_err += leaf.error
        return float(total_err)/len(self.leaves)

    def predict_obj(self):
        predict_obj = np.zeros(self.size)
        for leaf in self.leaves:
            predict = leaf.predict
            if predict == 1:
                print "PREDICT IS ONE"
                print sum(leaf.presence)
                for i in range(0, len(leaf.presence)):
                    if leaf.presence[i] == 1:
                        predict_obj[i] = predict
        return predict_obj











