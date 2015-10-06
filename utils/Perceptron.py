__author__ = 'Allison MacLeay'

import numpy as np
import CS6140_A_MacLeay.utils.Stats as mystats
from CS6140_A_MacLeay.utils import check_binary
from CS6140_A_MacLeay.utils.Stats import get_error
import pandas as pd


class Perceptron:
    def __init__(self, data, predict_col, learning_rate, max_iterations=1000):
        self.model = train_perceptron(data, predict_col, learning_rate, max_iterations)
        self.result_is_binary = check_binary(data[predict_col])
        self.training_score = self._get_score_from_data(data, data[predict_col])
        self.predict_column = predict_col

    def get_predicted(self, data):
        #print len(data.columns)
        #print len(self.model)
        #if self.predict_column not in data.columns:
        #    data[self.predict_column] = np.ones(len(data))
        predicted = np.dot(data, self.model)
        return predicted

    def get_score(self, predict, truth_set):
        return get_error(predict, truth_set, self.result_is_binary)

    def _get_score_from_data(self, data, truth_set):
        predicted = self.get_predicted(data)
        return self.get_score(predicted, truth_set)

    def print_score(self, score=None):
        caption = 'MSE: '
        if self.result_is_binary:
            caption = 'Accuracy: '
        if score is None:
            score = str(self.training_score)
        print '{} {}'.format(caption, score)







def train_perceptron(data, predict, learning_rate, max_iterations=1000):
    ct_i = 0
    size = len(data)
    cols = []
    for col in data.columns:
        if col != predict:
            cols.append(col)
    X = data[cols]

    # Add column of ones
    X['ones'] = np.ones(size)
    X = X.reindex()
    p = data[predict]

    # keep track of the mistakes
    last_m = 10000000000000

    # Switch x values from positive to negative if y < 0
    ct_neg_1 = 0
    print p[:5]
    for i, row in enumerate(X.iterrows()):
        if list(p)[i] < 0:
            ct_neg_1 += 1
            for cn, col in enumerate(X.columns):
                X.iloc[i, cn] *= -1

    #print 'ct neg is {} '.format(ct_neg_1)
    #print size

    # Get random array of w values
    w = mystats.init_w(5)[0]
    print 'w array'
    print w.head(5)
    print X.head(5)

    while ct_i < max_iterations:  # for each iteration

        J = []
        n_row = 0
        mistakes = pd.DataFrame(columns=X.columns)
        mistakes_x_sum = 0
        print 'w'
        print w
        for r_ct, row in X.iterrows():  # for each row
            x_sum = 0
            #print 'ct_i {} j {} w {} x {} x_sum {}'.format(ct_i, n_row, wj, X.iloc[n_row][ct_i], x_sum)
            for c_ct, col in enumerate(X.columns):
                x_sum += w[c_ct] * row[col]
            #if n_row < 5:
            #    print x_sum
            n_row += 1
            J.append(x_sum)
            if x_sum < 0:
                mistakes.loc[len(mistakes)] = row
                mistakes_x_sum += x_sum
                #print 'mistakes len {}'.format(len(mistakes))
                #print mistakes.head(5)

        # Check dot product (paranoia)
        #print 'J'
        #print J[:5]
        #print 'dot'
        #print np.dot(X,w)[:5]

        # check objective
        print 'sum of J is {}'.format(sum(J))
        print 'iteration: {} length: {} sum: {}'.format(ct_i, len(mistakes), -1 * mistakes_x_sum)

        print '{} mis*lr={}'.format(mistakes_x_sum, mistakes_x_sum * learning_rate)

        # update w
        for wi, wcol in enumerate(mistakes.columns):
            # Add the sum of mistakes for each column to w for that column
            w[wi] += learning_rate * sum(mistakes[wcol])
            print 'wcol: {} {}'.format(wcol, sum(mistakes[wcol]))
        #w += sum(mistakes) * learning_rate

        if last_m < (-1 * mistakes_x_sum):
            print 'last_m is {} and size of mistakes is {}'.format(last_m, -1 * mistakes_x_sum)
            break

        last_m = -1 * mistakes_x_sum
        ct_i += 1

    print pd.DataFrame(J).head(5)
    return w
