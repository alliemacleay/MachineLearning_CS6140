__author__ = 'Allison MacLeay'

from CS6140_A_MacLeay import utils
import numpy as np
import pandas as pd
import CS6140_A_MacLeay.utils.Stats as mystats

class Tree(object):
    def __init__(self):
        self.presence_array = None
        self.converged = False
        self.model = []
        self.training_info = []

    def fit(self, X, y, max_depth=3):
        loop_count = 0
        self.presence_array = np.ones(len(X))
        self.possible_thresholds = self.get_initial_thresholds(X)
        head_branch = Branch(X, y, self.presence_array)
        self.grow_tree(head_branch, X, y, max_depth)


    def grow_tree(self, head, X, y, max_depth):
        this_branch = head.split_branch(X, y)
        this_branch.split(X, y)
        stump = DecisionStump(this_branch.feature, this_branch.threshold)
        self.model.append(stump)
        if max_depth > 0 and not this_branch.leaf:
            left = this_branch.left_child
            right = this_branch.right_child

            max_depth -= 1
            self.grow_tree(left, X, y, max_depth)
            self.grow_tree(right, X, y, max_depth)



class Branch(object):
    def __init__(self, X, y, presence_array, parent_branch=None, theta=0.1):
        self.presence_array = presence_array
        self.theta = theta
        self.data_subset, self.truth_subset = self.get_subset(X, y)
        self.feature = None
        self.threshold = None
        self.unique = 100000
        self.info_gained = 0
        self.parent = parent_branch
        self.left_child = None
        self.right_child = None
        self.leaf = False
        self.info_gain = 0
        self.update_leaf_status()

    def update_leaf_status(self):
        if len(set(self.data_subset[0])) < 2:
            self.leaf = True
        if self.info_gain < self.theta:
            self.leaf = True


    def split_branch(self, X, y):
        self.feature, self.info_gain, self.threshold = self.choose_best_feature()
        self.update_leaf_status()
        presence_A, presence_B = self.split_presence_array(X, self.feature, self.threshold)
        self.left_child = Branch(X, presence_A, parent_branch=self)
        self.right_child = Branch(X, presence_B, parent_branch=self)
        if self.info_gain == 0:
            self.leaf = True

    def split_presence_array(self, X, column, threshold):
        array_l = []
        array_r = []
        by_col = utils.transpose_array(X)
        data = by_col[column]
        for i in range(len(data)):
            if data[i] > threshold:
                array_l.append(0)
                array_r.append(1)
            else:
                array_l.append(1)
                array_r.append(0)
        return array_l, array_r




    def choose_best_feature(self):
        by_col = utils.transpose_array(self.data_subset)
        max_info_gain = 0
        best_col = None
        col_threshold = None
        for j in range(len(by_col)):
            col, info_gain, threshold = self.compute_info_gain(by_col[j], self.truth_subset)
            if info_gain > max_info_gain:
                best_col = col
                max_info_gain = info_gain
                col_threshold = threshold
        return best_col, max_info_gain, col_threshold

    def compute_info_gain(self, column, y):
        subB = []
        truthB = []
        subC = []
        truthC = []
        split = randomSplit(column)
        for i in range(len(column)):  # bestSplit(column)
            if column[i] < split:
                subB.append(column[i])
                truthB.append(y[i])
            else:
                subC.append(column[i])
                truthC.append(y[i])

        info_gain = binary_entropy(y) - binary_entropy(truthB) + binary_entropy(truthC)
        #print 'Information Gain: %s' % info_gain
        return info_gain


    def get_subset(self, data, y):
        subset = []
        truth = []
        for i in range(len(data)):
            if self.presence_array[i] == 1:
                subset.append(data[i])
                truth.append(y[i])
        return subset, truth


def binary_entropy(truth):
    """ H(q) = SUMclases(P(class==1) * log2P(class==1))"""
    prob = float(truth.count(1))/len(truth)
    return np.log2(prob) * prob

def randomSplit(array):
    data = np.array(array)
    min = data.min()
    max = data.max()
    dat_range= max - min
    interv = dat_range/len(data)
    return np.random.random() * interv + min


class TreeOptimal(Tree):
    def __init__(self):
        super(TreeOptimal, self).__init__()
        self.possible_thresholds = None

    def get_initial_thresholds(self, data):
        by_col = utils.transpose_array(data)
        thresholds = []
        start = 100
        for j in range(len(by_col)):
            col_thresholds = []
            feature_j = [float(i) for i in np.array(by_col[j])]
            values = list(set(feature_j))
            values.sort()
            col_thresholds.append(values[0] - .01)
            for i in range(1, len(values)):
                mid = (values[i] - values[i-1])/2
                col_thresholds.append(values[i-1] + mid)
            col_thresholds.append(values[-1] + .01)
            thresholds.append(col_thresholds)
        return thresholds

class DecisionStump(object):
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = int(threshold)




