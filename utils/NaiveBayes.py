__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import CS6140_A_MacLeay.utils as utils

class NaiveBayes():
    def __init__(self, model_type, alpha=1):
        self.model_type = model_type
        self.train_acc = 0
        self.test_acc = 0
        self.model = []
        self.alpha = alpha  # smoothing parameter
        self.data_length = 0

    def train(self, data_rows, truth):
        if self.model_type == 0:
            self.data_length = len(data_rows)
            model = self.model_average_train(data_rows, truth)
            self.update_model(model)

    def predict(self, data_rows):
        prediction = []
        if self.model_type == 0:
            prediction = self.model_average_predict(data_rows)
        return prediction

    def update_model(self, model):
        """ if model exists update model by taking
        the average of old and new probabilities"""
        if len(self.model) == 0:
            self.model = add_smoothing(model, self.alpha, self.data_length)
        else:
            new_model = []
            model = add_smoothing(model, self.alpha, self.data_length)
            p1, p0, px, py = self.model
            for pi, p in enumerate([p1, p0, px]):
                new_p = model[pi]
                updated_p = []
                for xi in range(len(p)):  # cols
                    old_px = p[xi]
                    new_px = new_p[xi]
                    updated_p.append((old_px + new_px)/2)
                new_model.append(updated_p)
            new_model.append((py + model[3])/2)
            self.model = new_model

    def model_average_train(self, data_row, truth):
        """ return [prob_over_given_1, prob_over_given_0, prob_y1]
        prob_over_give_x = col1[mu, var, proabality], colx[mu, var, prob] ...
        """
        mus = hw3.get_mus(data_row)
        is_not_spam = hw3.get_sub_at_value(data_row, truth, 0)
        is_spam = hw3.get_sub_at_value(data_row, truth, 1)
        prob_over = get_prob_over(data_row, mus)
        prob_over_given_1 = get_prob_over(is_spam, mus)
        prob_over_given_0 = get_prob_over(is_not_spam, mus)
        l0 = len(prob_over_given_0)
        l1 = len(prob_over_given_1)
        if l1 != l0:
            addx = abs(l1-l0)
            fake_row = [0 for _ in range(addx)]
            if l1 > l0:
                prob_over_given_0 = fake_row
            else:
                prob_over_given_1 = fake_row
        prob_y1 = float(sum(truth))/len(truth)

        return [prob_over_given_1, prob_over_given_0, prob_over, prob_y1]

    def model_average_predict(self, data_row):
        """  For each row calculate the probability
        that y is 1 and the probability that y is 0
        P(Y|X) = ( P(X|Y) * P(Y) ) / ( P(X) )
        P(X) = prob_over (probability that x is above average for column)
        P(X|Y) = prob_over_given_c (probability that x is above average when y = c for column)
        P(Y) = prob_y ( probability of y )
        """
        mus = hw3.get_mus(data_row)
        data_cols = hw3.transpose_array(data_row)
        prob_over_given_1 = self.model[0]
        prob_over_given_0 = self.model[1]
        prob_over = self.model[2]
        prob_y1 = self.model[3]
        predict = []
        for r in range(len(data_row)):
            row = data_row[r]
            prob_1 = 1
            prob_0 = 1
            for c in range(len(row)):
                mu = mus[c]
                if row[c] > mu:
                    prob_x1 = prob_over_given_1[c]
                    prob_x0 = prob_over_given_0[c]
                    prob_xover = prob_over[c]
                else:
                    prob_x1 = 1 - prob_over_given_1[c]
                    prob_x0 = 1 - prob_over_given_0[c]
                    prob_xover = 1 - prob_over[c]
                prob_1 = prob_1 * prob_x1 * prob_y1 / prob_xover  #P(X|Y) * P(Y)
                prob_0 = prob_0 * prob_x0 * (1-prob_y1) / prob_xover
            if prob_1 > prob_0:
                predict.append(1)
            else:
                predict.append(0)
        return predict



def get_prob_over(data_by_row, mus):
    """
    Return array of arrays
    column[i] = [probability_above]
    """
    probability_above_mu = []
    size = len(data_by_row)
    by_col = hw3.transpose_array(data_by_row)
    for col in range(len(by_col)):
        total_over = 0
        column = by_col[col]
        mu_col = mus[col]
        var_col = utils.variance(by_col[col], size)
        for row in range(len(column)):
            if column[row] > mu_col:
                total_over += 1
        probability_above_mu.append(float(total_over)/size)
    return probability_above_mu

def calc_bayes(prob_x_given_y, mux, varx, prob_y, prob_x):
    return 0

def add_smoothing(array, alpha=0, length=1):
    p1 = array[0]
    p0 = array[1]
    px = array[2]
    py = array[3]
    for p in [p1, p0, px]:
        for i in range(len(p)):
            p[i] = (p[i] + alpha)/(length + alpha * len(p))
    return [p1, p0, px, py]



