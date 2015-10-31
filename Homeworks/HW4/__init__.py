__author__ = 'Allison MacLeay'

from CS6140_A_MacLeay import utils
import numpy as np
import pandas as pd
import CS6140_A_MacLeay.utils.Stats as mystats

class Tree(object):
    def __init__(self, max_depth=3):
        self.presence_array = None
        self.converged = False
        self.model = []
        self.training_info = []
        self.head = None
        self.leaves = []
        self.max_depth = max_depth
        self.weights = None

    def fit(self, X, y, d=None):  # d is weights
        self.presence_array = np.ones(len(X))
        if d is None:
            d = np.ones(len(X))
        self.weights = d
        self.possible_thresholds = self.get_initial_thresholds(X)
        self.head = self.initialize_branch(X, y)
        self.grow_tree(self.head, X, y, self.max_depth)
        self.print_tree()

    def predict(self, X):
        self.head.predict(X)
        predict_array = np.ones(len(X))
        for leaf in self.leaves:
            predicted = 1 if leaf.probability > .5 else -1
            for i in range(len(leaf.presence_array)):
                if leaf.presence_array[i] == 1:
                    predict_array[i] = predicted
        return predict_array

    def initialize_branch(self, X, y):
        raise NotImplementedError

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




    def grow_tree(self, this_branch, X, y, max_depth):
        if self.weights is None:
            self.weights = np.ones(len(X))
        this_branch.split_branch(X, y)
        stump = DecisionStump(this_branch.feature, this_branch.threshold)
        self.model.append(stump)
        if max_depth <= 1:
            this_branch.converged = True
        left = this_branch.left_child
        right = this_branch.right_child
        if not this_branch.converged:
            max_depth -= 1
            self.grow_tree(left, X, y, max_depth)
            self.grow_tree(right, X, y, max_depth)
        else:
            if left is not None:
                left.make_leaf(X, y)
                self.leaves.append(left)
            if right is not None:
                right.make_leaf(X, y)
                self.leaves.append(right)

    def print_tree(self):
        self.head.print_branch(True)






class Branch(object):
    def __init__(self, X, y, presence_array, level=1, parent_branch=None, theta=0.01, weights=None):
        if weights is None:
            weights = np.ones(len(X))
        self.presence_array = presence_array
        self.theta = theta
        self.data_subset, self.truth_subset, self.weights_subset = self.get_subset(X, y, weights)
        self.feature = None
        self.threshold = None
        self.unique = 100000
        self.parent = parent_branch
        self.left_child = None
        self.right_child = None
        self.converged = False
        self.info_gain = 0
        self.level = level
        self.leaf = False
        self.probability = None
        self.entropy = binary_entropy(self.truth_subset)

    def get_stump(self):
        return DecisionStump(self.feature, self.threshold)

    def predict(self, X):
        self.set_arrays(X)

    def set_arrays(self, X):
        left_X, right_X = self.split_presence_array(X, self.feature, self.threshold)
        if len(left_X) > 0 and self.left_child is not None and not self.left_child.leaf:
            self.left_child.presence_array = left_X
            self.left_child.set_arrays(X)
        if len(right_X) > 0 and self.right_child is not None and not self.right_child.leaf:
            self.right_child.presence_array = right_X
            self.right_child.set_arrays(X)

    def predict_split(self, X):
        presence_A, presence_B = self.split_presence_array(X, self.feature, self.threshold)
        if not self.converged:
            self.left_child = Branch(X, y, presence_A, self.level+1, parent_branch=self)
            self.right_child = Branch(X, y, presence_B, self.level+1, parent_branch=self)


    def print_branch(self, recursive=True):
        text = ''
        if self.converged:
            text = 'Last split: '
        if self.leaf:
            text = 'LEAF P {} N {} '.format(self.probability, sum(self.presence_array))
        text += 'Level {}: feature: {}  threshold: {} entropy: {} info gain: {}'.format(self.level, self.feature, self.threshold, self.entropy, self.info_gain)
        print text
        if self.right_child is not None:
            self.right_child.print_branch(recursive)
        if self.left_child is not None:
            self.left_child.print_branch(recursive)

    def make_leaf(self, X, y):
        self.leaf = True
        _, truth, _ = self.get_subset(X, y, None)
        self.probability = float(truth.count(1))/len(truth)
        #self.probability = float(y.count(1))/len(y)

    def update_leaf_status(self):
        if len(set(self.data_subset[0])) < 2:
            self.converged = True
        if self.info_gain < self.theta:
            self.converged = True


    def split_branch(self, X, y):
        self.converged = False
        self.feature, self.info_gain, self.threshold = self.choose_best_feature()
        self.update_leaf_status()
        presence_A, presence_B = self.split_presence_array(X, self.feature, self.threshold)
        self.update_leaf_status()
        if not self.converged:
            self.left_child = self.add_branch(X, y, presence_A)
            self.right_child = self.add_branch(X, y, presence_B)

    def add_branch(self, X, y, presence_array):
        raise NotImplementedError



    def split_presence_array(self, X, column, threshold):
        array_l = []
        array_r = []
        by_col = utils.transpose_array(X)
        data = by_col[column]
        for i in range(len(data)):
            if self.presence_array[i] == 1:
                if data[i] > threshold:
                    array_l.append(0)
                    array_r.append(1)
                else:
                    array_l.append(1)
                    array_r.append(0)
            else:
                array_l.append(0)
                array_r.append(0)
        return array_l, array_r




    def choose_best_feature(self):
        by_col = utils.transpose_array(self.data_subset)
        max_info_gain = -1
        min_weighted_error = 1.5
        best_col = None
        col_threshold = None
        for j in range(len(by_col)):
            info_gain, threshold, weighted_error = self.compute_info_gain(by_col[j], self.truth_subset)
            #TODO - fix objective function so it is organized
            #if info_gain > max_info_gain:
            #    best_ig_col = j
            #    max_info_gain = info_gain
            #    col_threshold = threshold
            if weighted_error < min_weighted_error:
                best_col = j
                min_weighted_error = weighted_error
                max_info_gain = info_gain
                col_threshold = threshold
        if best_col is None:
            print "BEST COL is NONE"
            self.print_branch(False)

        return best_col, max_info_gain, col_threshold


    def compute_info_gain(self, column, y):
        theta = self.choose_theta(column, y)
        #theta = optimalSplit(column)
        entropy_after = get_split_info_gain(column, y, theta)
        weighted_error = get_split_error(column, y, theta, self.weights_subset)
        if self.entropy != binary_entropy(y):
            print 'FALSE'
        info_gain = self.entropy - entropy_after
        #print 'Information Gain: %s' % info_gain
        return info_gain, theta, weighted_error

    def get_distance_from_mean(self, column, y):
        theta = self.choose_theta(column, y)
        d_from_m = get_split_error(column, y, theata, self.weights_subset)

    def choose_theta(self, column, truth):
        raise NotImplementedError


    def get_subset(self, data, y, d=None):
        subset = []
        truth = []
        weights=[] if d is not None else None


        for i in range(len(data)):
            if self.presence_array[i] == 1:
                subset.append(data[i])
                truth.append(y[i])
                if weights is not None:
                    weights.append(d[i])
        return subset, truth, weights


def binary_entropy(truth):
    """ H(q) = SUMclases(P(class==1) * log2P(class==1))"""
    if len(truth) == 0:
        return 0
    prob = float(truth.count(1))/len(truth)
    return calc_entropy(prob)

def calc_entropy(prob):
    if prob == 0 or prob == 1:
        return 0
    return -np.log2(prob) * prob - (np.log2(1-prob) * (1-prob))

def get_split_info_gain(column, y, theta):
    subB = []
    truthB = []
    subC = []
    truthC = []
    for i in range(len(column)):  # bestSplit(column)
        if column[i] < theta:
            subB.append(column[i])
            truthB.append(y[i])
        else:
            subC.append(column[i])
            truthC.append(y[i])
    return (float(len(truthB))/len(y) * binary_entropy(truthB)) + (float(len(truthC))/len(y) * binary_entropy(truthC))

def get_split_error(column, y, theta, d):
    weights = np.ones(len(column))
    sumd = sum(d)
    for i in range(len(d)):
        weights[i] = float(d[i])/sumd
    truthB = []
    truthC = []
    dB = []
    dC = []
    error = 0
    for i in range(len(column)):  # bestSplit(column)
        if column[i] < theta:
            truthB.append(y[i])
            dB.append(weights[i])
        else:
            truthC.append(y[i])
            dC.append(weights[i])
    prob_B = float(truthB.count(1))/len(truthB) if len(truthB) > 0 else 0
    prob_C = float(truthC.count(1))/len(truthC) if len(truthC) > 0 else 0
    pred_B = 1 if prob_B >= .5 else -1
    pred_C = 1 if prob_C >= .5 else -1
    for j in range(len(truthB)):
        if truthB[j] != pred_B:
            error += dB[j]
    for j in range(len(truthC)):
        if truthC[j] != pred_C:
            error += dC[j]
    return error

def randomSplit(array):
    data = np.array(array)
    min = data.min()
    max = data.max()
    dat_range= max - min
    interv = dat_range/len(data)
    return np.random.random() * interv + min


class TreeOptimal(Tree):
    def __init__(self, max_depth=3):
        super(TreeOptimal, self).__init__(max_depth)
        self.possible_thresholds = None

    def initialize_branch(self, X, y):
        return BranchOptimal(X, y, self.presence_array, weights=self.weights)


class BranchOptimal(Branch):
    def choose_theta(self, column, truth):
        feature_y = zip(column, truth)
        feature_y.sort()
        last = feature_y[0][0]
        num_a = 1
        num_b = len(column) - 1
        num_ones_a = 1 if feature_y[0][1] == 1 else 0
        num_ones_b = truth.count(1) - num_ones_a
        best_val = 3 #random initialization > 2
        best_i = None
        for i in range(1, len(column) - 1):
            if feature_y[i][1] == 1:
                num_ones_a += 1
                num_ones_b -= 1
            num_a += 1
            num_b -= 1
            if feature_y[i][0] == last:
                continue
            last = feature_y[i][0]
            perc_a = float(num_a)/len(column)
            perc_b = float(num_b)/len(column)
            prob_a = float(num_ones_a)/num_a
            prob_b = float(num_ones_b)/num_b
            new_val = perc_a * calc_entropy(prob_a) + perc_b * calc_entropy(prob_b)
            if new_val < best_val:
                best_val = new_val
                best_i = i
        if best_i is None:
            best_i = 0  # only 1 unique element
        mid = (feature_y[best_i + 1][0] - feature_y[best_i][0]) * .5
        return feature_y[best_i][0] + mid

    def add_branch(self, X, y, presence_A):
        return BranchOptimal(X, y, presence_A, self.level+1, parent_branch=self)

class TreeRandom(Tree):
    def __init__(self, max_depth=3):
        super(TreeRandom, self).__init__(max_depth)
        self.possible_thresholds = None

    def initialize_branch(self, X, y):
        return BranchRandom(X, y, self.presence_array, weights=self.weights)

class BranchRandom(Branch):
    def choose_theta(self, column, truth):
        return randomSplit(column)

    def add_branch(self, X, y, presence_A):
        return BranchRandom(X, y, presence_A, self.level+1, parent_branch=self)


class DecisionStump(object):
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = float(threshold)



def split_truth_from_data(data):
    """ Assumes that the truth column is the last column """
    truth_rows = utils.transpose_array(data)[-1]  # truth is by row
    data_rows = utils.transpose_array(utils.transpose_array(data)[:-1])  # data is by column
    for i in range(len(truth_rows)):
        if truth_rows[i] == 0:
            truth_rows[i] = -1
    return truth_rows, data_rows




