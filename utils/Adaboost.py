__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.utils.Stats as mystats
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import sklearn
#import sklearn.ensemble.AdaBoostRegressor as skada


def what():
    clf = skada()


class Adaboost(object):
    def __init__(self, max_rounds, learners=None):
        default_learner = 'SCDecisionTree'
        if learners == None:
            learners = [default_learner]
        self.training_errors = {}
        self.errors = {}
        self.errors_weighted = {}
        self.weight_distribution = {}
        self.total_weighted_errors = {}
        self.decision_stump_errors = {}  # local round errors
        self.thresholds = {}
        self.weights = {}
        self.average_weighted_errors = {}
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
        for round_number in self.errors.keys():
            print 'Round Number: {}'.format(round_number)
            print 'testing error: {} total weighted err (==1): {}\nweight distr {}\naverage weighted error: {}' \
                  ''.format(self.errors[round_number], self.total_weighted_errors[round_number],
                            self.weight_distribution[round_number], self.average_weighted_errors[round_number])

    def get_boost_round(self, round_number):
        raise NotImplementedError()

    def run(self, data):
        """ pass data into adaboost """
        round_number = 0
        weights = [1./len(data) for _ in range(len(data))]
        while round_number < self.max_rounds and not self.converged:
            self.weights[round_number + 1] = weights
            round = self.get_boost_round(round_number)
            round.run(data, weights)
            self.errors[round_number + 1] = round.error
            self.errors_weighted[round_number + 1] = round.errors_weighted
            self.weight_distribution[round_number + 1] = round.weight_distribution
            self.total_weighted_errors[round_number + 1] = round.total_weighted_error
            self.average_weighted_errors[round_number + 1] = round.average_weighted_error
            self.converged = round.converged
            weights = round.weight_distribution


            round_number += 1

    def get_decision_stump(self, predicted, truth):
        raise NotImplementedError


class BoostRound():
    def __init__(self, adaboost, round_number):
        self.learner = adaboost.learners[round_number]
        self.error = 1
        self.errors_weighted = []
        self.weight_distribution = []  # Dt(x)
        self.total_weighted_error = .5  # epsilon t
        self.err_matrix = []
        self.alpha = 0  # alpha t
        self.converged = False
        self.average_weighted_error = None
        self.stump = None

    def run(self, data, weights):
        truth, f_data = split_truth_from_data(data)
        model = self.fit(f_data, truth)
        predicted = self.predict(model, f_data)  # {-1, 1}
        #self.stump = self.get_decision_stump(predicted, truth)
        # Error matrix for round computed from test data
        self.err_matrix = self.compute_error_matrix(truth, predicted)
        self.error = self.get_error(self.err_matrix)  # 1 if correct, else 0
        self.errors_weighted = self.weight_errors(self.err_matrix, weights)
        self.set_weight_distribution_and_total()  # Dt(x) and epsilon
        self.set_alpha()



    def fit(self, data, truth):
        model = ''
        # create decision stumps
        model = DecisionTreeRegressor(max_depth=3)
        model.fit(data, truth)
        return model

    def predict(self, model, data):
        #  {-1, 1}
        predicted = model.predict(data)
        for i in range(len(predicted)):
            if predicted[i] > 0:
                predicted[i] = 1
            else:
                predicted[i] = -1
        return predicted
        #return self.test_predict(model, data)

    def test_predict(self, model, data):
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
            weighted.append(weights[i] if err_matrix[i]==0 else 0)
        return weighted

    def set_weight_distribution_and_total(self):
        sum_weights = sum(self.errors_weighted)
        #TODO - because python is messed up the sum of weights may not be exact
        # because of floating point stuff
        if sum_weights == 0:
            self.converged = True
            self.total_weighted_error = 0.
        else:
            self.average_weighted_error = float(sum_weights)/len(self.errors_weighted)
            self.weight_distribution = [float(i)/sum_weights for i in self.errors_weighted]
            self.total_weighted_error = sum(self.weight_distribution)

    def set_alpha(self):
        if self.total_weighted_error == 0:
            self.alpha = 'ERR: divide by zero'
        else:
            self.alpha = .5 * np.log( (1 - self.total_weighted_error) / self.total_weighted_error)



class AdaboostRandom(Adaboost):
    """ choose random thetas
    """
    def get_boost_round(self, round_number):
        return BoostRoundRandom(self, round_number)

class BoostRoundRandom(BoostRound):
    def get_decision_stump(self, predicted, truth):
        raise NotImplementedError


class AdaboostOptimal(Adaboost):
    """ choose optimal thetas
    """
    def get_boost_round(self, round_number):
        return BoostRoundOptimal(self, round_number)

class BoostRoundOptimal(BoostRound):
    def get_decision_stump(self, predicted, truth):
        pass





def split_truth_from_data(data):
    """ Assumes that the truth column is the last column """
    truth_rows = utils.transpose_array(data)[-1]  # truth is by row
    data_rows = utils.transpose_array(utils.transpose_array(data)[:-1])  # data is by column
    for i in range(len(truth_rows)):
        if truth_rows[i] == 0:
            truth_rows[i] = -1
    return truth_rows, data_rows

