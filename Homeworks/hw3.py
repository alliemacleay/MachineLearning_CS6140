__author__ = 'Allison MacLeay'

"""
Issues -
Q1
---------

Q2
_________
really bad accuracy again.  Check algorithm
Smoothing may have made it worse


"""

import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import CS6140_A_MacLeay.utils.NaiveBayes as nb
import CS6140_A_MacLeay.utils.Stats as mystats


def q1():
    """GDA """
    """Run the Gaussian Discriminant Analysis on the spambase data. Use the k-folds from the previous problem (1 for testing, k-1 for training, for each fold)
Since you have 57 real value features, each of the  2gaussians (for + class and for - class) will have a mean  vector with 57 components, and a they will have
either a common (shared) covariance matrix size 57x57. This covariance is estimated from all training data (both classes)
or two separate covariance 57x57 matrices (estimated separately for each class)
(you can use a Matlab or Python of Java built in function to estimated covariance matrices, but the estimator is easy to code up).
Looking at the training and testing performance, does it appear that the gaussian assumption (normal distributed data) holds for this particular dataset?
"""

    spamData = hw3.pandas_to_data(hw3.load_and_normalize_spambase())  # returns an array of arrays - this is by row
    k = 10
    train_acc_sum = 0
    k_folds = hw3.partition_folds(spamData, k)
    gda = hw3.GDA()
    for ki in range(k - 1):
        subset = []
        X, truth = hw3.separate_X_and_y(k_folds[ki])
        for y in [0, 1]:
            subset.append(hw3.get_sub_at_value(k_folds[ki], truth, y))
            truth_rows, data_rows, data_mus, y_mu = get_data_and_mus(subset[y])
            covariance_matrix = hw3.get_covar(data_rows, truth_rows)
            gda.train(data_rows, data_mus, covariance_matrix, y)
        predictions = gda.predict(X)
        accuracy = mystats.get_error(predictions, truth, True)
        print_output(ki, accuracy)


def q2():
    spamData = hw3.pandas_to_data(hw3.load_and_normalize_spambase())
    k = 10
    train_acc_sum = 0
    k_folds = hw3.partition_folds(spamData, k)
    for model_type in range(1):  #range(3):
        nb_model = nb.NaiveBayes(model_type, alpha=.001)
        for ki in range(k - 1):
            truth_rows, data_rows, data_mus, y_mu = get_data_and_mus(k_folds[ki])
            nb_model.train(data_rows, truth_rows)
            predict = nb_model.predict(data_rows)
            accuracy = hw3.get_accuracy(predict, truth_rows)
            train_acc_sum += accuracy
            print_output(ki, accuracy)
        truth_rows, data_rows, data_mus, y_mu = get_data_and_mus(k_folds[k - 1])
        test_predict = nb_model.predict(data_rows)
        test_accuracy = hw3.get_accuracy(test_predict, truth_rows)
        print_test_output(test_accuracy, float(train_acc_sum)/(k-1))



            #print len(k_folds[0])
    truth_rows, data_rows, data_mus, y_mu = get_data_and_mus(spamData)



def q3():
    """ Submit as pdf
    """
    pass
def q4():
    pass
def q5():
    """Written"""
    pass

def get_data_and_mus(spamData):
    truth_rows = hw3.transpose_array(spamData)[-1]  # truth is by row
    data_rows = hw3.transpose_array(hw3.transpose_array(spamData)[:-1])  # data is by column
    data_mus = hw3.get_mus(data_rows)
    y_mu = utils.average(truth_rows)
    return truth_rows, data_rows, data_mus, y_mu

def print_output(fold, accuracy):
    print 'fold {} ACC: {}'.format(fold + 1, accuracy)

def print_test_output(test_acc, train_acc):
    print 'Testing accuracy: {}\nAverage training accuracy: {}'.format(test_acc, train_acc)

def q1_step1():
    pass
