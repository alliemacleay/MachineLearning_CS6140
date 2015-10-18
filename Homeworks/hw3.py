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
    gdas = []
    for ki in range(k - 1):
        subset = []
        gda = hw3.GDA()
        X, truth = hw3.separate_X_and_y(k_folds[ki])
        covariance_matrix = hw3.get_covar(X)
        gda.p_y = float(sum(truth)) / len(truth)
        gda.train(X, covariance_matrix, truth)
        predictions = gda.predict(X)
        #print predictions
        accuracy = mystats.get_error(predictions, truth, True)
        #gdas.append(gda)
        print_output(ki, accuracy)
        #print gda.prob
        gdas.append(gda)
    #agg_gda = hw3.GDA()
    #agg_gda.aggregate(gdas)
    #X, truth = hw3.separate_X_and_y(k_folds[k-1])
    #predictions = agg_gda.predict(X)
    #accuracy = mystats.get_error(predictions, truth, True)


def q2():
    spamData = hw3.pandas_to_data(hw3.load_and_normalize_spambase())
    k = 10
    train_acc_sum = 0
    k_folds = hw3.partition_folds(spamData, k)
    for model_type in [2]:  #range(3):
        nb_models = []
        for ki in range(k - 1):
            alpha = .001 if model_type==0 else 0
            nb_model = nb.NaiveBayes(model_type, alpha=alpha)
            truth_rows, data_rows, data_mus, y_mu = hw3.get_data_and_mus(k_folds[ki])
            nb_model.train(data_rows, truth_rows)
            predict = nb_model.predict(data_rows)
            print predict
            accuracy = hw3.get_accuracy(predict, truth_rows)
            train_acc_sum += accuracy
            print_output(ki, accuracy)
            nb_models.append(nb_model)
        nb_combined = nb.NaiveBayes(model_type, alpha=.001)
        if model_type < 2:
            nb_combined.aggregate_model(nb_models)
        else:
            nb_combined.aggregate_model3(nb_models)
        truth_rows, data_rows, data_mus, y_mu = hw3.get_data_and_mus(k_folds[k - 1])
        test_predict = nb_combined.predict(data_rows)
        test_accuracy = hw3.get_accuracy(test_predict, truth_rows)
        print_test_output(test_accuracy, float(train_acc_sum)/(k-1))



            #print len(k_folds[0])
    truth_rows, data_rows, data_mus, y_mu = hw3.get_data_and_mus(spamData)



def q3():
    """ Submit as pdf
    """
    pass
def q4():
    """
    :return: mean, cov_matrix (std_dev), number in class
    """
    for data_set in [2,3]:
        data = hw3.pandas_to_data(utils.load_gaussian(data_set))
        q4_result_print([1, 2, 3], ['cov_1', 'cov2', 'cov3'], 100)

def q4_result_print(means, covariances, number):
    for i in range(len(means)):
        print 'mean_{}: {}'.format(i, means[i])
        print 'cov_{}: {}'.format(i, covariances[i])
        print 'n{}: {}'.format(i, number)

def q5():
    """Written"""
    pass

def print_output(fold, accuracy):
    print 'fold {} ACC: {}'.format(fold + 1, accuracy)

def print_test_output(test_acc, train_acc):
    print 'Testing accuracy: {}\nAverage training accuracy: {}'.format(test_acc, train_acc)

def q1_step1():
    pass
