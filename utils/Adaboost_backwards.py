__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.utils.Stats as mystats
import numpy as np

class Adaboost(object):
    def __init__(self, max_rounds, learners=None):
        default_learner = 'SCDecisionTree'
        if learners == None:
            learners = [default_learner]
        self.training_errors = {}
        self.testing_errors = {}
        self.training_errors_weighted = {}
        self.weight_distribution = {}
        self.total_weighted_errors = {}
        self.decision_stump_errors = {}  # local round errors
        self.thresholds = {}
        self.weights = {}
        self.converged = False
        self.max_rounds = max_rounds
        self.learners = learners

        num_learners = len(self.learners)
        if num_learners < max_rounds:
            # scipy descision tree is default
            default = self.learners[-1] if len(self.learners) > 0 else default_learner  # sci py decision tree
            for i in range(max_rounds - num_learners):
                self.learners.append(default)
        print 'inited'


    def print_stats(self):
        for round_number in self.testing_errors.keys():
            print 'Round Number: {}'.format(round_number)
            print 'training errors:'
            for i in range(len(self.training_errors[round_number])):
                print '{} weighted: {}'.format(self.training_errors[round_number][i], self.training_errors_weighted[round_number][i])
            print 'testing error: {} weight distr {}'.format(self.testing_errors[round_number], self.weight_distribution[round_number])



    def run(self, data):
        """ pass data into adaboost """
        round_number = 0
        weights = [1./len(data) for _ in range(len(data))]
        while round_number < self.max_rounds and not self.converged:
            self.weights[round_number + 1] = weights
            round = BoostRound(self, round_number)
            round.run(data, weights)
            self.training_errors[round_number + 1] = round.training_errors
            self.testing_errors[round_number + 1] = round.testing_error
            self.training_errors_weighted[round_number + 1] = round.training_errors_weighted
            self.weight_distribution[round_number + 1] = round.weight_distribution
            self.converged = round.converged


            round_number += 1





class AdaboostRandom(Adaboost):
    """ choose random thetas
    """
    pass


class AdaboostOptimal(Adaboost):
    """ choose optimal thetas
    """
    pass


class BoostRound():
    def __init__(self, adaboost, round_number):
        self.learner = adaboost.learners[round_number]
        self.training_errors = []
        self.testing_error = .5
        self.number_k_folds = 10
        self.testing_errors_weighted = []
        self.training_errors_weighted = []
        self.weight_distribution = []  # Dt(x)
        self.total_weighted_error = .5  # epsilon t
        self.err_matrix = []
        self.alpha = 0  # alpha t
        self.converged = False

    def run(self, data, weights):
        k_folds = hw3.partition_folds(data, self.number_k_folds)
        for k in xrange(self.number_k_folds - 1):
            err_matrix = []
            fold = k_folds[k]
            truth, f_data = split_truth_from_data(fold)
            model = self.fit(f_data)
            predicted = self.predict(model, f_data)  # {-1, 1}
            err_matrix = self.compute_error_matrix(truth, predicted)
            self.training_errors.append(self.get_error(err_matrix))
            self.training_errors_weighted.append(sum(self.weight_errors(err_matrix, weights)))
        fold = k_folds[self.number_k_folds - 1]
        truth, f_data = utils.split_truth_from_data(fold)
        predicted = self.predict(model, f_data)
        # Error matrix for round computed from test data
        self.err_matrix = self.compute_error_matrix(truth, predicted)
        self.testing_error = self.get_error(self.err_matrix)
        self.testing_errors_weighted = self.weight_errors(self.err_matrix, weights)
        self.set_weight_distribution_and_total()  # Dt(x) and epsilon
        self.set_alpha()



    def fit(self, data):
        model = ''
        # create decision stumps
        return model

    def predict(self, model, data):
        #  {-1, 1}
        predicted = np.ones(len(data))
        for i in range(len(predicted)):
            predicted[i] = 1
        return predicted

    def compute_error_matrix(self, truth, predicted):
        """ returns {0, 1}
        """
        err_matrix = np.ones(len(truth))
        for i in range(len(truth)):
            if truth[i] != predicted[i]:
                err_matrix[i] = 0
        return err_matrix


    def get_error(self, err_matrix):
        return 1 - float(sum(err_matrix))/len(err_matrix)

    def weight_errors(self, err_matrix, weights):
        weighted = []
        # Error matrix is inverted
        for i in range(len(err_matrix)):
            weighted.append(weights[i] if err_matrix[i] is 0 else 0)
        return weighted

    def set_weight_distribution_and_total(self):
        sum_weights = sum(self.testing_errors_weighted)
        if sum_weights == 0:
            self.converged = True
            self.total_weighted_error = 0.
        else:
            self.weight_distribution = [float(i)/sum_weights for i in self.testing_errors_weighted]
            self.total_weighted_error = sum(self.weight_distribution)

    def set_alpha(self):
        if self.total_weighted_error == 0:
            self.alpha = 'ERR: divide by zero'
        else:
            self.alpha = .5 * np.log( (1 - self.total_weighted_error) / self.total_weighted_error)


def split_truth_from_data(data):
    """ Assumes that the truth column is the last column """
    truth_rows = utils.transpose_array(data)[-1]  # truth is by row
    data_rows = utils.transpose_array(utils.transpose_array(data)[:-1])  # data is by column
    for i in range(len(truth_rows)):
        if truth_rows[i] == 0:
            truth_rows[i] = -1
    return truth_rows, data_rows

