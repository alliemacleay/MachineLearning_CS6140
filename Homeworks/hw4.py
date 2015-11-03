from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor
import CS6140_A_MacLeay.utils.Stats as mystats

__author__ = 'Allison MacLeay'

"""
Decision stump - feature (fi) threshold (tij)  pair
    {1, -1} 1 if feature exceeds threshold, else -1
    sort by fi values
    remove dups

    Optimal stumps
    construct thresholds between all values including one under min
        and one over max

    Random stumps

    Find value that maximizes |1/2 - error|

    Report
        local round error
        weighted training round error
        weighted testing round error
        weighted test AUC

    Plot
        round error
        training and test error
        test AUC
"""

import CS6140_A_MacLeay.utils.Adaboost as adab
import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import CS6140_A_MacLeay.Homeworks.HW4 as hw4
import CS6140_A_MacLeay.Homeworks.HW4.plots as plt
import CS6140_A_MacLeay.Homeworks.HW4.data_load as dl
import CS6140_A_MacLeay.Homeworks.HW4.bagging as bag
import CS6140_A_MacLeay.utils.Adaboost_compare as adac
import CS6140_A_MacLeay.utils.GradientBoost as gradb
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np

import os
import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.utils.Stats as mystats

def q1():
    spamData = hw3.pandas_to_data(hw3.load_and_normalize_spambase())
    k = 10
    all_folds = hw3.partition_folds(spamData, k)
    tprs = []
    fprs = []
    for i in [0]: #range(len(all_folds)):
        kf_data, kf_test = dl.get_train_and_test(all_folds, i)
        y, X = hw4.split_truth_from_data(kf_data)
        y_test, X_test = hw4.split_truth_from_data(kf_test)
        adaboost = run_adaboost(X, y, X_test, y_test, i)
        predicted = adaboost.predict(X)
        print(roc_auc_score(y, predicted))
        for i in range(len(adaboost.snapshots)):
            round_number = i + 1
            ab = adaboost.snapshots[i]
            yt_pred = ab.predict(X_test)
            round_err = float(np.sum([1 if yt==yp else 0 for yt, yp in zip(yt_pred, y_test)]))/len(y_test)
            adaboost.adaboost_error_test[round_number] = round_err
        print predicted[:20]
        print y[:20]
        name = 'q1'
        directory = '/Users/Admin/Dropbox/ML/MachineLearning_CS6140/CS6140_A_MacLeay/Homeworks'
        path = os.path.join(directory, name + 'hw4errors.pdf')
        tterrpath = os.path.join(directory, name + 'hw4_errors_test_train.pdf')
        print path
        plt.Errors([adaboost.local_errors]).plot_all_errors(path)
        plt.Errors([adaboost.adaboost_error, adaboost.adaboost_error_test]).plot_all_errors(tterrpath)
        roc = plt.ROC()
        #roc.add_tpr_fpr_arrays(adaboost.tpr.values(), adaboost.fpr.values())
        get_tpr_fpr(adaboost, roc, X_test, y_test, 30)
        roc.plot_ROC(os.path.join(directory, name + 'hw4_roc.pdf'))

def run_adaboost(X, y, X_test=None, y_test=None, name='q1'):  #c% for fold sizes
    """data
       c is percentage for subset size
       name for plot files
    """
    adaboost = adac.AdaboostOptimal(100, learner=lambda: DecisionTreeClassifier(max_depth=1))
    #adaboost = adac.AdaboostOptimal(100, learner=lambda: DecisionTreeClassifier(max_depth=1, splitter="random"))
    adaboost.fit(X, y) #, X_test, y_test)
    adaboost.print_stats()
    # Compute train & test AUCs at each round
    #for idx, ab in enumerate(adaboost.snapshots):
    #    if ab is not None:
    #        print("round {}: Train AUC={:.3f}. Test AUC={:.3f}".format(idx + 1, roc_auc_score(y, ab.predict(X)),
    #                                                               roc_auc_score(y_test, ab.predict(X_test))))

    return adaboost


def get_tpr_fpr(model, plot, X, y, num_points):
    for ti in range(num_points + 2):
        theta = ti * 1./(num_points + 1)
        predict = model.predict(X, theta)
        plot.add_tp_tn(predict, y, theta)

def q2():
    """Boosting on UCI datasets"""
    #crx = dl.data_q3_crx()
    crx = dl.data_q3_vote()
    num_points = len(crx)
    for i in xrange(5, 85, 5):
        percent = float(i)/100
        all_folds = hw4.partition_folds(crx, percent)
        kf_train = all_folds[0]
        kf_test = all_folds[1]
        y, X = hw4.split_truth_from_data(kf_train)
        y_test, X_test = hw4.split_truth_from_data(kf_test)
        adaboost = run_adaboost(X, y, X_test, y_test, 'q2_crx')
        yt_pred = adaboost.predict(X_test)
        yt_pred = adaboost._check_y(yt_pred)
        y_test = adaboost._check_y(y_test)
        round_err = float(np.sum([1 if yt!=yp else 0 for yt, yp in zip(yt_pred, y_test)]))/len(y_test)
        last_round = adaboost.local_errors.keys()[-1]
        #print 'Error at {}%: Train: {} Test: {}'.format(percent, adaboost.adaboost_error[last_round], round_err)
        print 'Error at {}%: Test: {}'.format(percent, round_err)


def q3():
    """Run your code from PB1 on Spambase dataset to perform Active Learning.
    Specifically:
    - start with a training set of about 5% of the data (selected randomly)
    - iterate M episodes: train the Adaboost for T rounds; from the datapoints
      not in the training set, select the 2% ones that are closest to the
      separation surface (boosting score F(x) closest to ) and add these to the
      training set (with labels). Repeat until the size of the training set
      reaches 50% of the data.

    How is the performance improving with the training set increase? Compare the
    performance of the Adaboost algorithm on the c% randomly selected training set
    with c% actively-built training set for several values of c : 5, 10, 15, 20,
    30, 50.
    """
    spamData = hw3.pandas_to_data(hw3.load_and_normalize_spambase())
    percent = .05
    all_folds = hw4.partition_folds_q4(spamData, percent)
    kf_train = all_folds[0]
    kf_test = all_folds[1]
    left_over = all_folds[2]

    while len(kf_train) < len(spamData)/2:
        y, X = hw4.split_truth_from_data(kf_train)
        y_test, X_test = hw4.split_truth_from_data(kf_test)
        adaboost = run_adaboost(X, y, X_test, y_test, 'q2_crx')

        yt_pred = adaboost.predict(X_test)
        order = adaboost.rank(X_test)
        yt_pred = adaboost._check_y(yt_pred)
        y_test = adaboost._check_y(y_test)
        round_err = float(np.sum([1 if yt!=yp else 0 for yt, yp in zip(yt_pred, y_test)]))/len(y_test)

        print 'Error {}'.format(round_err)
        shift_number = int(len(order) * .02)  # number of items to switch into training set
        mask = []
        for i in xrange(shift_number):
            mask.append(order[i])
            kf_train.append(kf_test[order[i]])
        new_test = [kf_test[i] for i in range(len(kf_test)) if i not in mask]
        for i in xrange(len(mask)):
            new_test.append(left_over[i])
        left_over = left_over[len(mask):]
        kf_test = new_test[:]
        print 'test len {} train len {} leftover len {} shifting {}'.format(len(kf_test), len(kf_train), len(left_over), shift_number)




def q4():
    """
    ECOC
    """
    news, features = dl.data_q4()
    ft_index = dl.index_features(features)
    settings = dl.metadata_q4()
    print len(settings.keys())
    label_settings = dl.metadata_q4_labels()
    print len(label_settings.keys())
    feature_list = int(np.random.random(20) * 1754)
    sub, indeces = dl.get_data_with_ft(news, ft_index, feature_list)


def q6():
    """ Bagging - sample with replacement """
    spamData = hw3.pandas_to_data(hw3.load_and_normalize_spambase())
    y, X = hw4.split_truth_from_data(spamData)
    bagged = bag.Bagging(max_rounds=100, sample_size=1000, learner=lambda: DecisionTreeClassifier(max_depth=3))
    bagged.fit(X, y)
    kf_fold = hw4.partition_folds(spamData, .4)
    test_y, test_X = hw4.split_truth_from_data(kf_fold[0])
    test_pred = bagged.predict(test_X)
    test_y = bagged._check_y(test_y)
    test_pred = bagged._check_y(test_pred)
    test_error = float(sum([0 if py == ty else 1 for py, ty in zip(test_pred, test_y)]))/len(test_y)
    print 'Final testing error: {}'.format(test_error)

def q7():
    h_test, h_train = utils.load_and_normalize_housing_set()
    housingData_test = hw3.pandas_to_data(h_test)
    housingData_train = hw3.pandas_to_data(h_train)
    y, X = hw4.split_truth_from_data(housingData_train)
    y_test, X_test = hw4.split_truth_from_data(housingData_test)
    #gb = GradientBoostingRegressor(learning_rate=.1, n_estimators=1, max_depth=1)
    gb = gradb.GradientBoostRegressor(learning_rate=.1, n_estimators=207, max_depth=1, learner=lambda: DecisionTreeRegressor(max_depth=1))
    gb.fit(X, y)
    gb.print_stats()
    yhat = gb.predict(X)
    print y[:10]
    print yhat[:10]
    print 'MSE: {}'.format(hw4.compute_mse(y, yhat))










