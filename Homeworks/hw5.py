__author__ = 'Allison MacLeay'

import os
from CS6140_A_MacLeay.utils.mnist import load_mnist
import CS6140_A_MacLeay.Homeworks.HW5 as hw5u
import CS6140_A_MacLeay.Homeworks.HW4.ecoc2 as ec
import CS6140_A_MacLeay.Homeworks.HW3 as hw3u
import CS6140_A_MacLeay.Homeworks.hw3 as hw3
import CS6140_A_MacLeay.Homeworks.HW4 as hw4u
import CS6140_A_MacLeay.Homeworks.hw4 as hw4
import CS6140_A_MacLeay.Homeworks.HW4.data_load as dl
import CS6140_A_MacLeay.utils.Adaboost_compare as adac
import CS6140_A_MacLeay.utils.NaiveBayes as nb
import CS6140_A_MacLeay.utils as utils
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression
import sklearn.linear_model as lm
import pandas as pd

"""
Homework 5
"""

def q1():
    """ feature analysis with Adaboost """
    #spamData = hw3u.pandas_to_data(hw3u.load_and_normalize_spambase())
    spamData = utils.load_and_normalize_polluted_spam_data()
    k = 10
    all_folds = hw3u.partition_folds(spamData, k)
    col_errs = []
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train)
    y_test, X_test = hw4u.split_truth_from_data(kf_test)

    # We're not actually cross-validating anything -- we just want feature weights
    #X = np.concatenate([X, X_test], axis=0)
    #y = np.concatenate([y, y_test], axis=0)

    #adaboost = adac.AdaboostOptimal(max_rounds=100, do_fast=False, learner=lambda: DecisionTreeClassifier(max_depth=1, splitter='random'))
    adaboost = adac.AdaboostOptimal(max_rounds=100, do_fast=False, learner=lambda: DecisionTreeClassifier(max_depth=1, splitter='best'))
    #adaboost = adac.AdaboostOptimal(max_rounds=10, do_fast=False, learner=hw4u.TreeOptimal)
    adaboost.fit(X, y)


    margin_fractions = get_margin_fractions(adaboost, X[0])
    #margin_fractions_v = hw5u.get_margin_fractions_validate(adaboost, X, y)
    #print col_errs
    ranked = rank(margin_fractions)
    print_ranks(ranked)

    pred = adaboost.predict(X_test)
    print 'Accuracy: {}'.format(accuracy_score(adaboost._check_y_not_zero(y_test), adaboost._check_y_not_zero(pred)))



    #ranked_v = rank(margin_fractions_v)
    #print_ranks(ranked_v)




def get_margin_fractions(ada, c):
    totals = sum(ada.margins) #* len(ada.stump)
    print 'total mine: {}'.format(totals)
    margin_fraction = []
    fmap = {}
    for i, s in enumerate(ada.stump):
        if s.feature not in fmap.keys():
            fmap[s.feature] = []
        fmap[s.feature].append(i)
    for f in range(len(c)):
        if f in fmap.keys():
            fmarg = 0
            rounds_used = fmap[f]
            for r in rounds_used:
                fmarg += ada.margins[r]
            margin_fraction.append(float(fmarg))
        else:
            margin_fraction.append(0.)
    return margin_fraction / totals



def rank(values):
    delta = zip(values, range(len(values)))
    delta.sort()
    return zip(*delta)

def print_ranks(ranked):
    print 'Rank from highest fraction to lowest (feature is 0 indexed)'
    for r in range(len(ranked[0]) - 1, -1, -1):
        print '{} frac of mgn: {} feature: {}'.format(len(ranked[0]) - r, list(ranked[0])[r], list(ranked[1])[r])

def q2():  # Done
    """
    standard deviation for some columns is 0
    """
    data = utils.load_and_normalize_polluted_spam_data()
    #data = utils.pandas_to_data(utils.load_and_normalize_spam_data())
    #data, _ = utils.random_sample(data, None, 300)  #TODO - remove
    #GaussianNB(data)
    GaussianNB(data, num_features=100)



def GaussianNB(X, num_features=None):
    model_type = 1
    train_acc_sum = 0
    test_acc_sum = 0
    k = 10
    nb_models = []
    if num_features is not None:
        y, X = utils.split_truth_from_data(X)
        q4_slct = SelectKBest(k=num_features).fit(X, y)
        X = q4_slct.transform(X)
        X = utils.add_row(X, y)
    k_folds = hw3u.partition_folds(X, k)
    for ki in range(k):
        grouped_fold = hw5u.group_fold(k_folds, ki)
        alpha = .001 if model_type==0 else 0
        mask_cols = check_cols(grouped_fold)
        #nb_model = nb.NaiveBayes(model_type, alpha=alpha, ignore_cols=mask_cols)
        nb_model = BernoulliNB()
        print 'len of kfolds {}'.format(len(grouped_fold))
        #truth_rows, data_rows, data_mus, y_mu = hw3u.get_data_and_mus(grouped_fold)
        truth_rows, data_rows = utils.split_truth_from_data(grouped_fold)
        print 'len of data {}'.format(len(data_rows))
        #nb_model.train(data_rows, truth_rows)
        nb_model.fit(data_rows, truth_rows)
        predict = nb_model.predict(data_rows)
        #print predict
        accuracy = hw3u.get_accuracy(predict, truth_rows)
        train_acc_sum += accuracy
        print_output(ki, accuracy)
        nb_models.append(nb_model)

        truth_rows, data_rows = utils.split_truth_from_data(k_folds[ki])
        test_predict = nb_model.predict(data_rows)
        test_accuracy = hw3u.get_accuracy(test_predict, truth_rows)
        test_acc_sum += test_accuracy
        print_output(ki, test_accuracy, 'test')

    print_test_output(float(train_acc_sum)/k, float(test_acc_sum)/k)

def check_cols(X):
    by_col = utils.transpose_array(X)
    msk = []
    for c in range(len(by_col)):
        col = by_col[c]
        if np.std(col) == 0:
            #print col
            #print '{} std_dev is 0'.format(c)
            msk.append(c)
    return msk

def print_test_output(train_acc, test_acc):
    print 'Average training accuracy: {}\n Average testing accuracy: {}\n'.format(train_acc, test_acc)

def print_output(fold, accuracy, ptype='train'):
    print 'fold {} {} ACC: {}'.format(fold + 1, ptype, accuracy)

def q3():  # Got points off b/c I have 89 accuracy instead of 92
    """ Logistic Regression """
    data = utils.load_and_normalize_polluted_spam_data()
    k = 10
    k_folds = hw3u.partition_folds(data, k)
    train_acc = []
    test_acc = []
    hw2_train_acc = []
    hw2_test_acc = []
    for ki in range(k):
        grouped_fold = hw5u.group_fold(k_folds, ki)
        y, X = utils.split_truth_from_data(grouped_fold)
        y_truth, X_test = utils.split_truth_from_data(k_folds[ki])
        clf = lm.LogisticRegression() #penalty="l1")
        ridge_clf = hw5u.Ridge()
        #clf = lm.Lasso(alpha=.5)
        #clf = lm.RidgeClassifier(alpha=.1)
        clf.fit(X, y)
        ridge_clf.fit(X, y)

        y_train = [1 if p >= .5 else 0 for p in clf.predict(X)]
        y_test = [1 if p >= .5 else 0 for p in clf.predict(X_test)]
        yhat_ridge_train = [1 if p >= .5 else 0 for p in ridge_clf.predict(X)]
        yhat_ridge_test = [1 if p >= .5 else 0 for p in ridge_clf.predict(X_test)]
        train_acc.append(accuracy_score(y, y_train))
        test_acc.append(accuracy_score(y_truth, y_test))
        hw2_train_acc.append(accuracy_score(y, yhat_ridge_train))
        hw2_test_acc.append(accuracy_score(y_truth, yhat_ridge_test))
        print 'Fold {} train acc: {} test acc: {} HW2 ridge train: {}  HW2 ridge test: {}'.format(ki+1, train_acc[-1], test_acc[-1], hw2_train_acc[-1], hw2_test_acc[-1])
    print 'Average acc - Train: {}  Test: {}  HW2 ridge: {}'.format(np.mean(train_acc), np.mean(test_acc), np.mean(hw2_train_acc), np.mean(hw2_test_acc))





def q4():  # fixed - need to demo
    """ spambase 20% Missing values """
    data = utils.load_and_fill_missing_spam_data('train')
    GaussianNB(data)

def q5():
    """ ECOC for image analysis
    1000 Set: train. Accuracy: 1.000
         Set: test. Accuracy: 0.851
    12,000 (20% of 60,000)
         Set: train. Accuracy: 0.923
         Set: test. Accuracy: 0.905

Process finished with exit code 0
    http://colah.github.io/posts/2014-10-Visualizing-MNIST/
    """
    path = os.path.join(os.getcwd(), 'data/HW5/haar')
    limit = 12000 #60,000
    images, labels = load_mnist('training', path=path)
    images /= 128.0
    X = []
    print 'processing images'
    black = [hw5u.count_black(b) for b in images[:limit]]
    #bdf = [pd.DataFrame(bd) for bd in black]
    #with open('save_img_' + str(limit) + '.csv', 'w') as fimg:
    #    pd.concat(bdf, axis=1).to_csv(fimg)
    print 'finished processing'

    rects = hw5u.get_rect_coords(100)
    #hw5u.show_rectangles(rects)

    for i in range(len(black)):
        row = []
        for r in range(len(rects)):
            h_diff, v_diff = hw5u.get_features(black[i], rects[r])
            row.append(h_diff)
            row.append(v_diff)
        X.append(row)
    save(X)
    # Each image is a row in table X.
    # Features are
    # rectangle_1_horizontal_difference, rectangle_1_vertical_difference, rectangle_2_ho...

    data = utils.add_row(X, labels)
    data_split = hw5u.split_test_and_train(data, .2)
    data_test = data_split[0]
    data_train = data_split[1]

    y_train, X_train = utils.split_truth_from_data(data_train)
    y_test, X_test = utils.split_truth_from_data(data_test)

    cls = ec.ECOCClassifier(learner=lambda: adac.AdaboostOptimal(learner=lambda: DecisionTreeClassifier(max_depth=1), max_rounds=200), #LogisticRegression,  # TODO: replace with AdaBoost
    #cls = ec.ECOCClassifier(learner=LogisticRegression,  # TODO: replace with AdaBoost
                         verbose=True,
                         encoding_type='exhaustive').fit(X_train, y_train)
    for set_name, X, y in [('train', X_train, y_train),
                       ('test', X_test, y_test)]:
        print("Set: {}. Accuracy: {:.3f}".format(set_name, accuracy_score(y, cls.predict(X))))


def save(data):
    df = pd.DataFrame(data)
    #with open('df_save_img_everything.csv', 'w') as fimg:
    with open('df_save_X.csv', 'w') as fimg:
        df.to_csv(fimg)


