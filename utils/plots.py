__author__ = 'Allison MacLeay'

import matplotlib.pyplot as plt

def points(array):
    plt_range = get_axes(array)
    print plt_range
    #plt.scatter(array, 'ro')
    plt.scatter(array[0], array[1])
    #plt.axis(plt_range)
    plt.show()

def get_axes(array):
    x_max = max(array[0])
    x_min = min(array[0])
    print array[1]
    y_max = max(array[1])
    y_min = min(array[1])
    return [x_min, x_max, y_min, y_max]

def fit_v_point(array):
    """ array[0] = x
        array[1] = y
        array[2] = linear fit"""
    plt.scatter(array[0], array[1], color='red')
    plt.scatter(array[0], array[2], color='blue')
    plt.show()
