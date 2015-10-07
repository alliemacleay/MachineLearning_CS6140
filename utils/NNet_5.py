# -*- coding: utf-8 -*-


from __future__ import unicode_literals
#from __future__ import print_function

import numpy as np


class Neuron(object):
    def __init__(self, n_inputs):
        self.w = np.random.random(size=n_inputs)
        self.bias = np.random.random()
        self.output_val = None
        self.err = None

    def get_output(self, input_vec):
        sum_val = np.sum([wi * xi for wi, xi in zip(self.w, input_vec)]) + self.bias
        return 1.0/(1.0 + np.exp(-sum_val))

class InputNeuron(object):
    def __init__(self):
        pass

    def get_output(self, input_vec):
        return input_vec

class BackpropNN(object):
    def __init__(self, layers):
        self.layers = layers
        self.input_units = [InputNeuron() for _ in range(self.layers[0])]
        self.hidden_units = [Neuron(self.layers[0]) for _ in range(self.layers[1])]
        self.output_units = [Neuron(self.layers[1]) for _ in range(self.layers[2])]

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        # Inputs and outputs from every unit in every layer of the network
        for epoch in range(epochs):
            if epoch % 100 == 0:
                print 'epoch: {}'.format(epoch)

            for input_vec, output_vec in zip(X, y):

                for input_idx in range(self.layers[0]):
                    for hidden_i in self.hidden_units:
                        hidden_i.output_val = hidden_i.get_output(input_vec)
                        #print(hidden_i)

                    for output_i, truth in zip(self.output_units, output_vec):
                        output_i.output_val = output_i.get_output(
                            [hidden_i.output_val for hidden_i in self.hidden_units])
                        output_i.err = output_i.output_val * (1.0 - output_i.output_val) * (truth - output_i.output_val)

                    for idx, hidden_i in enumerate(self.hidden_units):
                        hidden_i.err = hidden_i.output_val * (1.0 - hidden_i.output_val) * \
                            np.sum([out_unit.err * out_unit.w[idx] for out_unit in self.output_units])

                    for unit in self.hidden_units:
                        for j, wj in enumerate(unit.w):
                            unit.w[j] += learning_rate * unit.err * input_vec[j]


                        unit.bias += learning_rate * unit.err

                    for unit in self.output_units:
                        for j, wj in enumerate(unit.w):
                            unit.w[j] += learning_rate * unit.err * self.hidden_units[j].output_val
                            #print('j is %s' % str(j))
                            #print 'hidden {}'.format(self.hidden_units[j].output_val)

                        unit.bias += learning_rate * unit.err

    def predict(self, input_vec):
        for hidden in self.hidden_units:
            hidden.output_val = hidden.get_output(input_vec)
            #print 'hidden: {} '.format(hidden.output_val)
        for output in self.output_units:
            output.output_val = output.get_output([hidden.output_val for hidden in self.hidden_units])

        return [output.output_val for output in self.output_units]


def run_autoencoder(learning_rate, epochs):
    nn = BackpropNN([8,3,8])
    X = np.zeros(shape=(8, 8))
    for j in range(X.shape[1]):
        X[j, j] = 1

    nn.fit(X, X, learning_rate=learning_rate, epochs=epochs)

    for input_vec in X:
        print input_vec, np.round(nn.predict(input_vec), 1)
        #print input_vec
        #print nn.predict(input_vec)

def run():
    np.random.seed(0xC0FFEE)
    run_autoencoder(0.4, 100)

if __name__ == '__main__':
    run()

