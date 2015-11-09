__author__ = 'Allison MacLeay'

import os
from CS6140_A_MacLeay.utils.mnist import load_mnist
import CS6140_A_MacLeay.Homeworks.HW5 as hw5u
import CS6140_A_MacLeay.Homeworks.HW3 as hw3u
import CS6140_A_MacLeay.Homeworks.hw3 as hw3
import CS6140_A_MacLeay.Homeworks.HW4 as hw4u
import CS6140_A_MacLeay.Homeworks.hw4 as hw4
import CS6140_A_MacLeay.Homeworks.HW4.data_load as dl
import CS6140_A_MacLeay.utils.Adaboost_compare as adac
import CS6140_A_MacLeay.utils.NaiveBayes as nb
import CS6140_A_MacLeay.utils as utils
import numpy as np
import pandas as pd

"""
Homework 5
"""

def q1():
    """ feature analysis with Adaboost """
    spamData = hw3u.pandas_to_data(hw3u.load_and_normalize_spambase())
    k = 10
    all_folds = hw3u.partition_folds(spamData, k)
    margins = []
    abs_sum_alpha = 0
    col_errs = []
    for i in [0]:  #range(k):
        kf_train, kf_test = dl.get_train_and_test(all_folds, i)
        y, X_full = hw4u.split_truth_from_data(kf_train)
        y_test, X_test_full = hw4u.split_truth_from_data(kf_test)
        for coln in range(len(X_full[0])):
            X = hw5u.remove_col(X_full, coln)
            X_test = hw5u.remove_col(X_test_full, coln)
            adaboost = adac.AdaboostOptimal(max_rounds=300)
            adaboost.fit(X, y)
            yt_pred = adaboost.predict(X_test)
            yt_pred = adaboost._check_y(yt_pred)
            y_test = adaboost._check_y(y_test)
            round_err = float(np.sum([1 if yt!=yp else 0 for yt, yp in zip(yt_pred, y_test)]))/len(y_test)
            margins.append(adaboost.feature_margin)
            last_round = adaboost.local_errors.keys()[-1]
            abs_sum_alpha += adaboost.abs_sum_alpha
            #print 'Error at {}%: Train: {} Test: {}'.format(percent, adaboost.adaboost_error[last_round], round_err)
            print 'Error at {}%: Margin: {}'.format(round_err, adaboost.feature_margin)
            col_errs.append(round_err)

    total_margin = float(sum(margins)) / abs_sum_alpha
    print 'Total Margin: {}'.format(total_margin)
    print col_errs


def q2():
    """
    standard deviation for some columns is 0
    """
    data = utils.load_and_normalize_polluted_spam_data()
    data, _ = utils.random_sample(data, None, 20)
    GaussianNB(data)


def GaussianNB(X, features='all'):
    model_type = 1
    train_acc_sum = 0
    k = 10
    nb_models = []
    k_folds = hw3u.partition_folds(X, k)
    for ki in range(k - 1):
        alpha = .001 if model_type==0 else 0
        mask_cols = check_cols(k_folds[ki])
        nb_model = nb.NaiveBayes(model_type, alpha=alpha, ignore_cols=mask_cols)
        print 'len of kfolds {}'.format(len(k_folds[ki]))
        truth_rows, data_rows, data_mus, y_mu = hw3u.get_data_and_mus(k_folds[ki])
        print 'len of data {}'.format(data_rows)
        nb_model.train(data_rows, truth_rows)
        predict = nb_model.predict(data_rows)
        print predict
        accuracy = hw3u.get_accuracy(predict, truth_rows)
        train_acc_sum += accuracy
        print_output(ki, accuracy)
        nb_models.append(nb_model)
    nb_combined = nb.NaiveBayes(model_type, alpha=.001)
    if model_type < 2:
        nb_combined.aggregate_model(nb_models)
    else:
        nb_combined.aggregate_model3(nb_models)
    print 'len of kfolds {}'.format(len(k_folds[k - 1]))
    truth_rows, data_rows, data_mus, y_mu = hw3u.get_data_and_mus(k_folds[k - 1])
    test_predict = nb_combined.predict(data_rows)
    test_accuracy = hw3.get_accuracy(test_predict, truth_rows)
    print_test_output(test_accuracy, float(train_acc_sum)/(k-1))

def check_cols(X):
    by_col = utils.transpose_array(X)
    msk = []
    for c in range(len(by_col)):
        col = by_col[c]
        if np.std(col) == 0:
            print col
            print '{} std_dev is 0'.format(c)
            msk.append(c)
    return msk

def print_test_output(test_acc, train_acc):
    print 'Average training accuracy: {}\nTesting accuracy: {}\n'.format(train_acc, test_acc)

def print_output(fold, accuracy):
    print 'fold {} ACC: {}'.format(fold + 1, accuracy)

def q3():
    """"""
    pass



def q4():
    """"""
    pass

def q5():
    """ ECOC for image analysis"""
    path = os.path.join(os.getcwd(), 'data/HW5/haar')
    images, labels = load_mnist('training', path=path)
    images /= 128.0
    X = []
    print 'processing images'
    black = [hw5u.count_black(b) for b in images[:10]]
    print 'finished processing'
    #bdf = [pd.DataFrame(bd) for bd in black]
    #with open('save_img.csv', 'w') as fimg:
    #    pd.concat(bdf, axis=1).to_csv(fimg)

    rects = hw5u.get_rect_coords(200)
    for i in range(len(black)):
        for r in range(len(rects)):
            h_diff, v_diff = hw5u.get_features(black[i], rects[r])
        X.append([h_diff, v_diff])
    print X


