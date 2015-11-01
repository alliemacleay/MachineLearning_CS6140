from sklearn.metrics import roc_auc_score

__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import CS6140_A_MacLeay.Homeworks.HW4 as hw4
import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.utils.Stats as mystats
import numpy as np
import sklearn
#import sklearn.ensemble.AdaBoostRegressor as skada
import CS6140_A_MacLeay.utils.AdaboostRound as adar


class Adaboost(object):
    def __init__(self, max_rounds, learners=None):
        default_learner = 'SCDecisionTree'
        if learners == None:
            learners = [default_learner]
        self.local_errors = {}
        self.adaboost_error = {}
        self.adaboost_error_test = {}
        self.weight_distribution = {}
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
        self.auc = {}

        num_learners = len(self.learners)
        if num_learners < max_rounds:
            # scipy descision tree is default
            default = self.learners[-1] if len(self.learners) > 0 else default_learner  # sci py decision tree
            for i in range(max_rounds - num_learners):
                self.learners.append(default)
        print 'inited'

    def print_stats(self):
        for round_number in self.local_errors.keys():
            print 'Round: {} Feature: {} Threshold: {} Round_err: {} Train_err: {} Test_err {} AUC {}' \
                  ''.format(round_number, self.decision_stumps[round_number].feature,
                            self.decision_stumps[round_number].threshold, self.local_errors[round_number],
                            self.adaboost_error[round_number],
                            self.adaboost_error_test[round_number], self.auc[round_number])


    def get_boost_round(self, round_number):
        raise NotImplementedError()

    def fit(self, data, y, test=None, test_y=None):
        """ pass data into adaboost """
        round_number = 0
        weights = [1./len(data) for _ in range(len(data))]
        while round_number < self.max_rounds and not self.converged:
            self.weights[round_number + 1] = weights
            round = self.get_boost_round(round_number)
            round.run(data, y, weights)
            self.local_errors[round_number + 1] = round.error
            self.weight_distribution[round_number + 1] = round.weight_distribution
            self.decision_stumps[round_number + 1] = round.stump
            self.alphas[round_number + 1] = round.alpha
            ds_predict = self.predict(data)
            self.adaboost_error[round_number + 1] = self.get_error(ds_predict, y)
            if test is not None and test_y is not None:
                ds_predict_test = self.predict(test)
                self.adaboost_error_test[round_number + 1] = self.get_error(ds_predict_test, test_y)
            else:
                self.adaboost_error_test[round_number + 1] = ''
            self.tpr[round_number + 1], self.fpr[round_number + 1] = self.get_tpr_fpr(ds_predict, y)
            self.auc[round_number + 1] = roc_auc_score(y, ds_predict)
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

    def get_error(self, predict, truth):
        correct = 0
        for i, p in enumerate(predict):
            if p == truth[i]:
                correct += 1
        return 1 - float(correct)/len(truth)

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


class AdaboostOptimal(Adaboost):
    """ choose optimal thetas
    """
    def get_boost_round(self, round_number):
        return adar.BoostRoundOptimal(self, round_number)

class AdaboostRandom(Adaboost):
    """ choose random thetas
    """
    def get_boost_round(self, round_number):
        return adar.BoostRoundRandom(self, round_number)