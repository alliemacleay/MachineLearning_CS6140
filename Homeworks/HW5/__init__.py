__author__ = 'Allison MacLeay'

import numpy as np


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
    coords = []
    for n in xrange(number):
        height = np.random.randint(1, size/2) * 2
        width = np.random.randint(1, size/2) * 2
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

