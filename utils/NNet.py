__author__ = 'Allison MacLeay'

from copy import deepcopy
import numpy as np

"""
Neural Network

t - target - vector[k]
z - output - vector[k]
y - hidden - vector[j]
x - input  - vector[i]

wji - weight between x and y - matrix[i, j]
        wji[ 11 12 13 14 1j
             21 22 23 24 2j
             i1 i2 i3 i4 ij ]
wkj - weight between y and z - matrix[j, k]
        wkj[ 11 12 13 14 1k
             21 22 23 24 2k
             j1 j2 j3 j4 jk ]

wji_apx - wji approximation
wkj_apx - wkj approximation
outputs_apx - outputs approximation

theta - no idea - bias maybe?  init to 1...

i = 8
j = 3
k = 8

wji[8, 3]
wkj[3, 8]

"""

class NeuralNet():

    def __init__(self):
        self.wji, self.wkj = init_apx()
        self.inputs, self.hiddens, self.outputs = init_nnet()

    def get_wlayer_i(self, layer, i):
        if layer == 0:
            return self.get_wi(i)
        else:
            return self.get_wj(i)

    def get_wj(self, i):
        return self.wkj[i]


    def get_wi(self, j):
        arr = []
        for k in self.wji.keys():
            arr.append(self.wji[k][j])
        return arr

    def get_output(self, layer, i):
        output = []
        if layer == 0:
            # hidden
            output = self.hiddens[i]
        else:
            # final output
            output = self.outputs[i]
        return output

    def get_tuple(self, i):
        check_it(self.inputs, self.hiddens, self.outputs)
        return self.inputs[i], self.hiddens[i], self.outputs[i]



def init_nnet():
    # initialize inputs (x)
    zeros = [0, 0, 0, 0, 0, 0, 0, 0]
    inputs = {}

    for i in range(0, 8):
        in_array = zeros[:]
        in_array[i] = 1
        inputs[i] = in_array

    # initialize outputs (z)
    outputs = deepcopy(inputs)

    # initialize hidden weights (y)
    hiddens = {0: [.89, .04, .08],
               1: [.15, .99, .99],
               2: [.01, .97, .27],
               3: [.99, .97, .71],
               4: [.03, .05, .02],
               5: [.01, .11, .88],
               6: [.80, .01, .98],
               7: [.60, .94, .01]}
    check_it(inputs, hiddens, outputs)
    return inputs, hiddens, outputs



def check_it(inputs, hiddens, outputs):
    # check all this initialization
    print 'checking inputs'
    print inputs
    print outputs
    print hiddens

    # make sure python isn't stupid and was copied by value and not reference
    # except it is... so use slice notation [:] for array and dict.copy() for hash
    print 'checking pbv'
    inputs[4][0] = 999
    print inputs
    print outputs
    inputs[4][0] = 0


def init_apx():
    # Initialize all weights and biases in network
    # wji - matrix[8, 3]
    # wkj - matrix[3, 8]
    wji_apx = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    wkj_apx = {0: [], 1: [], 2: []}
    range_start = 0
    range_end = 1
    for i in range(0, 8):
        wji_apx[i] = random_array(range_start, range_end, 3)
    for i in range(0, 3):
        wkj_apx[i] = random_array(range_start, range_end, 8)
    return wji_apx, wkj_apx


def random_array(start, end, size):
    step = float(end - start) / size
    arr = []
    for i in range(0, size):
        arr.append(start + step * i)
    return arr


def init_theta(size):
    # theta should be 0 to start #TODO verify this
    arr = []
    for i in range(0, size):
        arr.append(0)
    return arr


def run_all(inputs, hiddens, outputs, num):  # num is just for testing.  should iterate through entire set
    # run NNet for num examples in training set
    layers = 2  # I will iterate through layers using layer
    nn = NeuralNet()
    wji_apx, wkj_apx = init_apx()
    theta = init_theta(num)
    sum_j = {}
    for i in range(0, num):
        #input, hidden, output = get_tuple(inputs, hiddens, outputs, i)
        input, hidden, output = nn.get_tuple(i)

        # This should happen for the entire set before this
        #wji, wkj = init_apx()

        for layer in range(layers):
            # propogate inputs forward
            O = nn.get_output(layer, i)
            wlayer = nn.get_wlayer_i(layer, i)  # this should return 8 weights
            print 'wi is {} length hidden is {}'.format(wlayer, len(hidden))
            for j in range(len(hidden)):
                print 'hidden: ' + str(hidden[j])
                print '{} * {}'.format(wlayer[j], O[j])
                #input[i] =
            sum_j[i] = 0

            # iterate through hidden layer
            for j in range(0, len(hidden)):
                sum_j[i] += wlayer[i] * O[j]

            print 'i is {}'.format(i)
            print sum_j

def run():
    inputs, hiddens, outputs = init_nnet()
    run_all(inputs, hiddens, outputs, 1)



