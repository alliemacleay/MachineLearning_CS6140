__author__ = 'Allison MacLeay'

from copy import deepcopy

import numpy as np

import CS6140_A_MacLeay.utils.NNet_5 as N5


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
        """

        :rtype : object
        """
        self.wji, self.wkj = init_apx()
        self.inputs, self.hiddens, self.outputs = init_nnet()
        self.learning_rate = .0005

    def get_wlayer_i(self, layer, i):
        if layer == 0:
            return self.get_wi(i)
        else:
            return self.get_wj(i)

    def get_wlayer(self, layer):
        if layer == 0:
            return self.wji
        else:
            return self.wkj

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

    def xget_tuple(self, i, j):
        """
        i(0-7) j(0-2) i(0-7)
        """
        check_it(self.inputs, self.hiddens, self.outputs)
        return self.inputs[i], self.hiddens[i], self.outputs[i]

    def update_weights(self, E, O):
        lwji = len(self.wji)
        lwkj = len(self.wkj)
        #for i in range(lwji):
        #    for k in range(lwkj):
        #        delta_ji = self.learning_rate * E[i] * O[k]



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
    theta = init_theta(layers)
    sum_j = {}

    #for i in range(0, num):
    for i in [0]:
        #input, hidden, output = get_tuple(inputs, hiddens, outputs, i)
        #input = []
        input, hidden, output = nn.get_tuple(i)


        # This should happen for the entire set before this
        #wji, wkj = init_apx()

        # initialize error matrix
        err = []
        for i in range(layers):
            err.append(0)

        for layer in range(layers):
            # propogate inputs forward
            O = nn.get_output(layer, i)[:]  # returns 3 for i=0 and 8 for i=1

            T = O[:]  #  target
            print 'length of O is {} hiddens[0] {}'.format(len(O), len(nn.hiddens[0]))
            wlayer = nn.get_wlayer(layer)  # this should return 8 weights for i=0
            o_length = len(O)
            for append_i in range(len(O), len(wlayer)):
                O.append(0)
            print 'wlayer is {} length wlayer is {}'.format(wlayer, len(wlayer))
            for j in range(len(wlayer)):
                wj = wlayer[j]
                sumk = 0
                for k in range(o_length):
                    print 'counter {}: {} += {} * {}'.format(k, sumk, O[k], wlayer[j][k])
                    sumk += O[k] * wlayer[j][k]
                input[j] = sumk + theta[layer]
                O[j] = float(1)/(1 + np.exp(-input[j]))
            err[layer] = []
            for j in range(o_length):
                print 'J IS {}'.format(j)
                err[layer].append(O[j] * (1-O[j]) * (T[j] - O[j]))
                layer_ct = layer + 1
                sum_layer = 0
                while layers + 1 > layer_ct < len(err):
                    if err[layer_ct] > 0:
                        weights = nn.get_wlayer(layer_ct)
                        for w_ct in range(len(weights)):
                            sum_layer += err[layer_ct] * weights[w_ct][j]
                    layer_ct += 1

                if sum_layer != 0:
                    err[layer] = O[j] * (1 - O[j]) * sum_layer

                #print 'layer {} new Oj = {}'.format(layer, O[j])

            print 'len O for layer=0 should be 8 layer {} len {}'.format(layer, len(O))
            print err[layer]
            print O
        # Outside layer loop
        nn.update_weights(err, O)

def xrun_all():
    layers = 2  # I will iterate through layers using layer
    nn = NeuralNet()
    wji_apx, wkj_apx = init_apx()
    theta = init_theta(layers)
    I = []
    O = []
    I[0] = nn.hiddens[0][0]

    # For each training tuple
    for j in range(layers):  # layer (0-1)
        I = nn.get_inputs(layer)

                #sum_i +=



def run():
    #inputs, hiddens, outputs = init_nnet()
    #run_all(inputs, hiddens, outputs, 1)
    #N3.run()
    #N2.run()
    #N4.run_autoencoder()
    N5.run()



