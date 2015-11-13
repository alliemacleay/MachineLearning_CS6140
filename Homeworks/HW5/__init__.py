__author__ = 'Allison MacLeay'

import numpy as np
import pandas as pd
import CS6140_A_MacLeay.Homeworks.hw2_new as hw2
import CS6140_A_MacLeay.utils.Stats as mystats
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def count_black(arr):
    #print len(arr)
    black = np.zeros(shape=(len(arr), len(arr[0])))
    sum = 0
    for i in range(len(arr)):
        sum_black = 0
        for j in range(len(arr[i])):
            if arr[i][j] == 1:
                sum_black += 1
            tot = (i + 1) * (j + 1)
            prev = black[i - 1][j] if i > 0 else 0
            black[i][j] = float(prev * (j + 1) * i + sum_black)/ tot
    return black


def get_rect_coords(number, size=28):
    # min area is 170
    coords = []
    for n in xrange(number):
        height = np.random.randint(5, size)
        minw = 130/height
        width = np.random.randint(minw, size)
        start_y = np.random.randint(0, size - height)
        start_x = np.random.randint(0, size - width)
        # return coordinate of the rectangles we need for area calculation, not the corners of the rectangle of interest
        c = [[start_x - 1, start_y - 1], [start_x + width, start_y - 1], [start_x - 1, height + start_y], [start_x + width, start_y + height]]
        coords.append(c)
    return coords


def get_features(barray, coords):
    total = get_black_amt(barray, coords)
    vsplit_x = (coords[1][0] - coords[0][0])/2 + coords[0][0]
    hsplit_y = (coords[2][1] - coords[0][1])/2 + coords[0][1]
    left = [coords[0], [vsplit_x, coords[1][1]], coords[2], [vsplit_x, coords[3][1]]]
    top = [coords[0], coords[1], [coords[2][0], hsplit_y], [coords[3][0], hsplit_y]]
    left_amt = get_black_amt(barray, left)
    top_amt = get_black_amt(barray, top)
    return total - top_amt, total - left_amt


def get_black_amt(barray, coords):
    """A = rect[0]
       B = rect[1]
       C = rect[2]
       D = rect[3]
    """
    #              B(x)  -  A(x)         *      C(y)     -     A(y)
    size = (coords[1][0] - coords[0][0]) * (coords[3][1] - coords[0][1])
    rect = [barray[c[1]][c[0]] * (c[1] + 1) * (c[0] + 1) for c in coords]
    #        D     +   A     -    B     +   C
    number_black = rect[3] + rect[0] - (rect[1] + rect[2])
    return float(number_black)/ size if number_black > 0 else 0

def remove_col(data, c):
    X = []
    for i in range(len(data)):
        X.append([])
        for j in range(len(data[0])):
            if j != c:
                X[i].append(data[i][j])

    return X

def group_fold(data, skip):
    group = []
    for i in range(len(data)):
        if i != skip:
            for r in range(len(data[i])):
                group.append(data[i][r])
    return group

class Ridge(object):
    """ I did hw2 in a strange way but it worked.
        This is a wrapper to use the linear ridge
        regression like scikit's implementation
    """
    def __init__(self):
        self.w = []

    def fit(self, X, y):
        df = pd.DataFrame(X)
        self.w = mystats.get_linridge_w(df, y, True)

    def predict(self, X):
        df = pd.DataFrame(X)
        return mystats.predict(df, self.w, True, [])



def lin_ridge_wrap(X, y):

    df = pd.DataFrame(X)
    df['is_spam'] = y
    accuracy = hw2.linear_reg(df, 'is_spam', True, True)  # returns accuracy
    return accuracy


def split_test_and_train(data, percentage):
    # split into test and train at a certain percentage
    num = int(len(data) * percentage)
    array = [[], []]
    idx_arr = np.arange(len(data))
    np.random.shuffle(idx_arr)
    setn = 0
    for i in xrange(len(data)):
        if i > num:
            setn = 1
        array[setn].append(data[idx_arr[i]])
    return array  # [ test_array, train_array ]

def show_rectangles(rects, fname='hw5_q5_rect.png'):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.set_ylim([0,28])
    ax2.set_xlim([0,28])
    for r in rects:
        corner = r[2]
        height = r[0][1] - r[2][1] + 1
        width = r[1][0] - r[0][0] + 1

        ax2.add_patch(
            patches.Rectangle(
            (corner[0], corner[1]),
            width,
            height,
            fill=False,      # remove background
            edgecolor=random_color()
            )
        )
    fig2.savefig(fname, dpi=90, bbox_inches='tight')


def random_color():
    r = np.random.random()
    g = np.random.random()
    b = np.random.random()
    return (r,g,b)




