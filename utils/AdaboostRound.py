__author__ = 'Allison MacLeay'


from sklearn.tree import DecisionTreeClassifier
import CS6140_A_MacLeay.Homeworks.HW4 as decTree
import CS6140_A_MacLeay.Homeworks.HW4 as hw4
import numpy as np

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
        last_alpha = self.alpha
        model = self.fit(f_data, truth, weights)
        predicted = self.predict(model, f_data)  # {-1, 1}
        #self.stump = self.get_decision_stump(predicted, truth)
        # Error matrix for round computed from test data
        self.err_matrix = self.compute_error_matrix(truth, predicted)
        self.error = self.get_error(self.err_matrix)  # 1 if correct, else 0
        self.errors_weighted = self.weight_errors(self.err_matrix, weights)
        self.set_alpha(weights)
        self.set_weight_distribution_and_total(weights)  # Dt(x) and epsilon

        if model.tree_.feature[0] < 0:
            raise ValueError('oops')
        self.stump = hw4.DecisionStump(model.tree_.feature[0], model.tree_.threshold[0])



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
        # Error matrix is inverted # 0 if error, 1 if correct
        for i in range(len(err_matrix)):
            weighted.append(weights[i] * np.exp(1 if err_matrix[i]==0 else -1))
        return weighted

    def set_weight_distribution_and_total(self, last_weights):
        sum_weights = sum(self.errors_weighted)
        #TODO - because python is messed up the sum of weights may not be exact
        # because of floating point stuff

        if sum_weights == 0:
            self.converged = True
            self.total_weighted_error = 0.
            self.average_weighted_error = 0.
        else:
            self.average_weighted_error = float(sum_weights)/len(self.errors_weighted)
            wd = [self.errors_weighted[i] * np.exp(-self.alpha if self.err_matrix[i]==1 else self.alpha)
                                        for i in range(len(self.errors_weighted))]
            sum_wd = sum(wd)
            self.weight_distribution = [float(w)/sum_wd for w in wd]

            if np.any(np.isnan(self.weight_distribution)):
                raise ValueError('nans in weights')

            self.total_weighted_error = sum(self.weight_distribution)

    def set_alpha(self, weights):
        #TODO fix alphas
        epsilon = self.get_epsilon(weights)
        if epsilon == 0 or epsilon >= 1:
            raise ValueError('oops')
        else:
            self.alpha = .5 * np.log( (1 - epsilon) / epsilon)

    def get_epsilon(self, weights):
        epsilon = 0
        for i, is_correct in enumerate(self.err_matrix):
            if is_correct==0:
                epsilon += weights[i]
        return float(epsilon)/len(self.errors_weighted)


class BoostRoundRandom(BoostRound):
    def fit(self, data, truth, weights):
        model = ''
        # create decision stumps
        print 'BoostRoundRandom'
        #model = DecisionTreeRegressor(max_depth=3)
        model = decTree.TreeRandom(max_depth=1)
        model.fit(data, truth, weights)
        return model



class BoostRoundOptimal(BoostRound):
    def fit(self, data, truth, weights):
        model = ''
        # create decision stumps
        print 'BoostRoundOptimal'
        model = DecisionTreeClassifier(max_depth=1)
        #model = decTree.TreeOptimal(max_depth=1)
        model.fit(data, truth, sample_weight=weights)
        return model





