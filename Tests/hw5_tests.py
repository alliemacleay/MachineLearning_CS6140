__author__ = 'Allison MacLeay'

from pylab import *
import numpy as np
from CS6140_A_MacLeay.utils.mnist import load_mnist
import CS6140_A_MacLeay.Homeworks.hw5 as hw5
import CS6140_A_MacLeay.Homeworks.HW5 as hw5u
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def test_mnist():
    path=os.path.join(os.getcwd(), 'data/HW5/haar')
    if not os.path.exists(path):
        print os.getcwd()
    images, labels = load_mnist('training', digits=[0], path=path)
    #print images
    imshow(images.mean(axis=0), cmap=cm.gray)
    show()

    images, labels = load_mnist("training", path=path)
    images /= 255.0
    #print images


def test_rectangle():
    path=os.path.join(os.getcwd(), 'data/HW5/haar')
    images, labels = load_mnist('training', digits=[4], path=path)
    one_img = images[7]
    one_img /= 128.0
    b = hw5u.count_black(one_img)
    print one_img
    print b


def test_count_black():
    t = [[0,0,0,0]]
    t.append([1,1,0,0])
    t.append([0,0,1,1])
    t.append([1,1,0,0])
    print t
    return hw5u.count_black(t)


def test_area():
    b = test_count_black()
    c = hw5u.get_rect_coords(1, size=len(b))
    #c = [[[1, -1], [2, -1], [1, 1], [2, 1]]]  # 0
    #c = [[[0, -1], [1, -1], [0, 1], [1, 1]]] # .5
    print b
    print c[0]
    print hw5u.get_black_amt(b, c[0])


def test_plot():
    rects = hw5u.get_rect_coords(10)
    hw5u.show_rectangles(rects, fname='test_rects.png')

def test_random_color():
    return hw5u.random_color()


def unit_tests():
    #test_mnist()
    #test_rectangle()
    #print test_count_black()
    test_plot()
    #print test_random_color()
    #print hw5u.get_rect_coords(5)
    #test_area()

if __name__ == '__main__':
    #unit_tests()
    #hw5.q1()
    #hw5.q2()  # full points
    #hw5.q3()  # points off
    #hw5.q4()
    hw5.q5()


