__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import CS6140_A_MacLeay.utils as utils
import numpy as np




class NaiveBayes():
    def __init__(self, model_type, alpha=1):
        self.model_type = model_type
        self.train_acc = 0
        self.test_acc = 0
        self.model = []
        self.alpha = alpha  # smoothing parameter
        self.data_length = 0
        self.cutoffs = None
        self.y_prob = None

    def train(self, data_rows, truth):
        self.data_length = len(data_rows)
        if self.model_type == 0:
            self.model = self.model_average_train(data_rows, truth)
        if self.model_type == 1:
            self.model = self.model_gaussian_rand_var_train(data_rows, truth)
        if self.model_type == 2:
            self.model = self.model_bin_train(data_rows, truth, 4)
        if self.model_type == 3:
            self.model = self.model_bin_train(data_rows, truth, 9)

    def predict(self, data_rows):
        prediction = []
        if self.model_type == 0:
            prediction = self.model_average_predict(data_rows)
        if self.model_type == 1:
            prediction = self.model_gaussian_rand_var_predict(data_rows)
        if self.model_type > 1:
            prediction = self.model_bin_predict(data_rows)
        return prediction


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
        self.y_prob = prob_y1

        return [prob_over_given_1, prob_over_given_0, prob_over, prob_y1]

    def model_bin_train(self, data_row, truth, num_bins=2):
        #TODO add epsilon
        model = {}
        dmat = np.matrix(data_row)
        drange = dmat.max() - dmat.min()
        bin_size = float(drange) / num_bins
        cutoffs = [dmat.min() + bin_size * i for i in range(num_bins)]
        #epsilon = float(alpha * 1) / len(covar_matrix)
        for label in [0,1]:
            # transpose to go by column
            sub_data = hw3.transpose_array(hw3.get_sub_at_value(data_row, truth, label))
            model[label] = hw3.bins_per_column(sub_data, cutoffs)
            # probability of bin given label
        self.y_prob = float(sum(truth))/len(truth)
        self.cutoffs = cutoffs
        return model

    def model_bin_predict(self, data_row):
        """
        probality[0] = [xlabel_0_prob, xlabel_1_prob, ..., xlabel_n_prob]
                        probability of y == 0 given xlabel
        probality[1] = [xlabel_0_prob, xlabel_1_prob, ..., xlabel_n_prob]
                        probability of y == 1 given xlabel
        """
        for r in range(len(data_row)):
            row = data_row[r]
            prob = [1 for _ in range(len(self.cutoffs))]
            probability = [[] for _ in [0, 1]]
            for c in range(len(row)):
                xbin = hw3.classify_x(row[c], self.cutoffs)
                for label in [0, 1]:
                    # model[0] = [col1: prob_bin1, prob_bin2 ...], [col2:...]
                    prob[label] = prob[label] * self.model[label][c][xbin]
                for label in [0, 1]:
                    prob_y = self.y_prob if label == 1 else 1 - self.y_prob
                    probability[label].append(prob[label] * prob_y)
            return self.nb_predict(probability)






    def model_gaussian_rand_var_train(self, data, truth):
        mus = {}
        std_dev = {}
        for label in [0,1]:
            sub_data = hw3.get_sub_at_value(data, truth, label)
            mus[label] = hw3.get_mus(sub_data)
            std_dev[label] = hw3.get_std_dev(sub_data)
        self.y_prob = float(sum(truth))/len(truth)
        return [mus, std_dev, float(sum(truth))/len(truth)]

    def model_gaussian_rand_var_predict(self, data):
        """ model = [[mus_by_col], [std_dev_by_col], prob_y]"""
        std_devs = self.model[1]
        mus = self.model[0]
        y_prob = self.model[2]
        probabilities = {}
        for label in [0, 1]:
            prob_of_y = y_prob if label==1 else (1-y_prob)
            probabilities[label] = hw3.univariate_normal(data, std_devs[label], mus[label], prob_of_y)

        return self.nb_predict(probabilities)


    def nb_predict(self, probabilities):
        """
        probality[0] = [xlabel_0_prob, xlabel_1_prob, ..., xlabel_n_prob]
                        probability of y == 0 given xlabel
        probality[1] = [xlabel_0_prob, xlabel_1_prob, ..., xlabel_n_prob]
                        probability of y == 1 given xlabel
        """
        predict = []
        for r in range(len(probabilities[0])):
            max_label = None
            for label in [0, 1]:
                if max_label == None:
                    max_label = [probabilities[label][r], label]
                if probabilities[label][r] > max_label[0]:
                    max_label = [probabilities[label][r], label]
            predict.append(max_label[1])
        return predict


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
                prob_1 = prob_1 * prob_x1 #* prob_y1 #/ prob_xover  #P(X|Y) * P(Y)
                prob_0 = prob_0 * prob_x0 #* (1-prob_y1) #/ prob_xover
                #prob_1 = prob_1 + np.log(prob_x1) + np.log(prob_y1)
                #prob_0 = prob_0 + np.log(prob_x0) + np.log(1-prob_y1)
            prob_1 = prob_1 * prob_y1
            prob_0 = prob_0 * (1 - prob_y1)
            if prob_1 > prob_0:
                predict.append(1)
            else:
                predict.append(0)
        return predict

    def aggregate_model(self, models):
        """ Average of all:
        [prob_over_given_1, prob_over_given_0, prob_over, prob_y1]
        """
        if self.model_type > 0:
            #TODO - this is Baaaaad
            self.aggregate_model2(models)
            return
        init = [0 for _ in models[0].model[0]]
        mult_fields = 3 if self.model_type == 0 else 2
        agg_model = []
        for i in range(mult_fields):
            agg_model.append(init[:])
        agg_model.append(0)
        total_models = len(models)
        for m in range(len(models)):
            model = models[m].model
            for i in range(mult_fields):
                probs = model[i]
                for c in range(len(probs)):  # columns
                    agg_model[i][c] += probs[c]
            agg_model[3] += model[3]
        for i in range(3):
            for c in range(len(probs)):  # columns
                agg_model[i][c] = float(agg_model[i][c])/total_models
        agg_model[3] = float(agg_model[3])/total_models
        self.model = agg_model

    def aggregate_model2(self, models):
        """ Average of all:
        [prob_of_y_given_x_and_1, prob_y_given_x_and_0, prob_y1]
        """
        print "AGG MOD2"
        # initiate models as {0: [0,0,0...len(cols)], 1: [0, 0 ,0, ..len(cols)]
        init = {i:[0 for _ in models[0].model[0][0]] for i in [0,1]}
        mult_fields = 3 if self.model_type == 0 else 2
        agg_model = []
        for i in range(mult_fields):
            agg_model.append(init)
        agg_model.append(0)
        total_models = len(models)
        for m in range(len(models)):
            model = models[m].model
            for i in range(mult_fields):
                probs = model[i]
                for label in range(len(probs)):
                    for c in range(len(probs[label])):
                        agg_model[i][label][c] += probs[label][c]
            agg_model[mult_fields] += model[mult_fields]
        for i in range(mult_fields):
            for c in range(len(probs[0])):  # columns
                for label in [0, 1]:
                    agg_model[i][label][c] = float(agg_model[i][label][c])/total_models
        agg_model[mult_fields] = float(agg_model[mult_fields])/total_models
        self.model = agg_model



    def aggregate_model3(self, models):
        """ Average of all:
        [prob_of_y_given_x_and_1, prob_y_given_x_and_0, prob_y1]
        """
        print "AGG MOD3"
        self.cutoffs = models[0].cutoffs
        # initiate models as {0: [0,0,0...len(cols)], 1: [0, 0 ,0, ..len(cols)]
        num_bins = len(self.cutoffs)
        zeros = np.zeros(num_bins)
        agg_model = {i:[zeros[:] for _ in models[0].model[0]] for i in [0, 1]}
        total_models = len(models)
        y_prob_sum = 0
        for m in range(len(models)):
            model = models[m].model
            for label in [0, 1]:
                probs = model[label]
                for c in range(len(probs)):
                    for xbin_i in range(len(probs[c])):
                        agg_model[label][c][xbin_i] += probs[c][xbin_i]
            y_prob_sum += models[m].y_prob
        for label in [0, 1]:
            for c in range(len(probs[0])):  # columns
                for xbin_i in range(len(probs[0])): # number of bins
                    agg_model[label][c][xbin_i] = float(agg_model[label][c][xbin_i])/total_models
        self.y_prob = float(y_prob_sum)/total_models
        self.model = agg_model


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

def add_smoothing(array, alpha=0.0, length=1):
    p1 = array[0]
    p0 = array[1]
    px = array[2]
    py = array[3]
    for p in [p1, p0, px]:
        for i in range(len(p)):
            p[i] = float(p[i] + alpha)/(length + alpha * len(p))
    return [p1, p0, px, py]



