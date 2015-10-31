__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import CS6140_A_MacLeay.Homeworks.HW4 as hw4
import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.utils.Stats as mystats
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import CS6140_A_MacLeay.Homeworks.HW4 as decTree
import sklearn
#import sklearn.ensemble.AdaBoostRegressor as skada


class Adaboost(object):
    def __init__(self, max_rounds, learners=None):
        default_learner = 'SCDecisionTree'
        if learners == None:
            learners = [default_learner]
        self.training_errors = {}
        self.errors = {}
        self.gamma_prod = {}
        self.local_errors = {}
        self.errors_weighted = {}  # TODO - remove
        self.weight_distribution = {}
        self.total_weighted_errors = {}  # TODO - remove
        self.decision_stump_errors = {}  # local round errors
        self.thresholds = {}
        self.weights = {}
        self.average_weighted_errors = {}
        self.converged = False
        self.max_rounds = max_rounds
        self.learners = learners
        self.decision_stumps = {}
        self.alphas = {}
        self.tpr = {}
        self.fpr = {}

        num_learners = len(self.learners)
        if num_learners < max_rounds:
            # scipy descision tree is default
            default = self.learners[-1] if len(self.learners) > 0 else default_learner  # sci py decision tree
            for i in range(max_rounds - num_learners):
                self.learners.append(default)
        print 'inited'

    def print_stats_q1(self):
        test_error = '?'
        auc = '?'
        for round_number in self.errors.keys():
            print 'Round: {} Feature: {} Threshold: {} Round_err: {} Train_err: {} Test_err {} AUC {}' \
                  ''.format(round_number, self.decision_stumps[round_number].feature,
                            self.decision_stumps[round_number].threshold,
                            self.decision_stump_errors[round_number], self.errors[round_number],
                            test_error, auc)

    def print_stats(self):
        for round_number in self.errors.keys():
            print 'Round Number: {}'.format(round_number)
            print 'testing error: {} total weighted err (==1): {}\nweight distr {}\naverage weighted error: {}' \
                  ''.format(self.errors[round_number], self.total_weighted_errors[round_number],
                            self.weight_distribution[round_number], self.average_weighted_errors[round_number])

    def get_boost_round(self, round_number):
        raise NotImplementedError()

    def fit(self, data, y):
        """ pass data into adaboost """
        round_number = 0
        weights = [1./len(data) for _ in range(len(data))]
        while round_number < self.max_rounds and not self.converged:
            self.weights[round_number + 1] = weights
            round = self.get_boost_round(round_number)
            round.run(data, y, weights)
            gamma = np.sqrt(round.error * (1 - round.error))
            self.local_errors[round_number + 1] = round.error
            self.gamma_prod[round_number + 1] = gamma if round_number==0 else gamma * self.gamma_prod[round_number]
            self.errors_weighted[round_number + 1] = round.errors_weighted
            self.weight_distribution[round_number + 1] = round.weight_distribution
            self.total_weighted_errors[round_number + 1] = round.total_weighted_error
            self.average_weighted_errors[round_number + 1] = round.average_weighted_error
            self.decision_stumps[round_number + 1] = round.stump
            self.alphas[round_number + 1] = round.alpha
            self.errors[round_number + 1] = 2 * self.gamma_prod[round_number + 1]
            ds_predict = self.predict(data)
            self.decision_stump_errors[round_number +1] = self.get_error(ds_predict, y)
            self.tpr[round_number + 1], self.fpr[round_number + 1] = self.get_tpr_fpr(ds_predict, y)
            self.converged = round.converged
            weights = round.weight_distribution

            round_number += 1

    def predict(self, data):
        predicted = []
        for row in data:
            sigma = 0
            alpha_sum = 0
            for round in self.decision_stumps.keys():
                feature = self.decision_stumps[round].feature
                threshold = self.decision_stumps[round].threshold
                alpha = self.alphas[round]
                ht = 1 if row[feature] >= threshold else 0
                sigma += alpha * ht
                alpha_sum += alpha
            predicted.append(1 if sigma >= .5*alpha_sum else -1)
        return predicted

    def get_error(self, f, y):
        num_correct = 0
        for i in range(len(f)):
            if f[i] == y[i]:
                num_correct += 1
        p_correct = float(num_correct)/len(f)
        return p_correct if p_correct >= 1 else 1-p_correct

    def get_tpr_fpr(self, f, y):
        sum_tpr = 0
        sum_fpr = 0
        for i in range(len(f)):
            if y[i] == 1:
                if f[i] == y[i]:
                    sum_tpr += 1
            else:
                if f[i] == y[i]:
                    sum_fpr += 1
        return float(sum_tpr)/len(f), float(sum_fpr)/len(f)


    def get_decision_stump(self, predicted, truth):
        raise NotImplementedError

    def get_h(self, column, threshold):
        count_ones = 0
        for i in column:
            if i > threshold:
                count_ones += 1
        return float(count_ones)/len(count_ones)


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

    def run(self, f_data, truth, weights):
        model = self.fit(f_data, truth, weights)
        predicted = self.predict(model, f_data)  # {-1, 1}
        #self.stump = self.get_decision_stump(predicted, truth)
        # Error matrix for round computed from test data
        self.err_matrix = self.compute_error_matrix(truth, predicted)
        self.error = self.get_error(self.err_matrix)  # 1 if correct, else 0
        self.errors_weighted = self.weight_errors(self.err_matrix, weights)
        self.set_weight_distribution_and_total()  # Dt(x) and epsilon
        self.set_alpha()
        self.stump = model.head.get_stump()



    def fit(self, data, truth, weights):
        raise NotImplementedError


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
            weighted.append(weights[i] if err_matrix[i]==0 else 1e-10)
        return weighted

    def set_weight_distribution_and_total(self):
        sum_weights = sum(self.errors_weighted)
        #TODO - because python is messed up the sum of weights may not be exact
        # because of floating point stuff
        if sum_weights == 0:
            self.converged = True
            self.total_weighted_error = 0.
            self.average_weighted_error = 0.
        else:
            self.average_weighted_error = float(sum_weights)/len(self.errors_weighted)
            self.weight_distribution = [float(i)/sum_weights for i in self.errors_weighted]
            self.total_weighted_error = sum(self.weight_distribution)

    def set_alpha(self):
        #TODO fix alphas
        epsilon = self.average_weighted_error
        if epsilon is None:
            self.alpha = np.nan
        elif epsilon == 0:
            self.alpha = 0 #'ERR: divide by zero'
        else:
            self.alpha = np.log( (1 - epsilon) / epsilon)



class AdaboostRandom(Adaboost):
    """ choose random thetas
    """
    def get_boost_round(self, round_number):
        return BoostRoundRandom(self, round_number)

class BoostRoundRandom(BoostRound):
    def fit(self, data, truth, weights):
        model = ''
        # create decision stumps
        print 'BoostRoundRandom'
        #model = DecisionTreeRegressor(max_depth=3)
        model = decTree.TreeRandom(max_depth=1)
        model.fit(data, truth, weights)
        return model


class AdaboostOptimal(Adaboost):
    """ choose optimal thetas
    """
    def get_boost_round(self, round_number):
        return BoostRoundOptimal(self, round_number)

class BoostRoundOptimal(BoostRound):
    def fit(self, data, truth, weights):
        model = ''
        # create decision stumps
        print 'BoostRoundOptimal'
        #model = DecisionTreeRegressor(max_depth=3)
        model = decTree.TreeOptimal(max_depth=1)
        model.fit(data, truth, weights)
        return model


