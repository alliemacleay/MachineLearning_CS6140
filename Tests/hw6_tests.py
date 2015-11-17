import CS6140_A_MacLeay.Homeworks.hw6 as hw6
import CS6140_A_MacLeay.Homeworks.HW6 as hw6u
import CS6140_A_MacLeay.utils as utils
import pandas as pd

__author__ = 'Allison MacLeay'

def do_tests():
    #test_mnist_load()
    test_mnist_load_small()


def test_mnist_load_small():
    X = hw6u.load_mnist_features(10)
    print X.shape


def test_mnist_load():
    data = pd.read_csv('df_save_img_everything.csv')
    print len(data)
    X = utils.pandas_to_data(data)
    print X[0]

if __name__ == '__main__':
    #do_tests()
    #hw6.q1a()
    hw6.q1b()
