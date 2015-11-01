from sklearn.metrics import roc_auc_score

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
        adaboost = adab.AdaboostOptimal(50)
        adaboost.fit(X, y, X_test, y_test)
        adaboost.print_stats()
        predicted = adaboost.predict(X)
        print(roc_auc_score(y, predicted))
        print predicted[:20]
        print y[:20]
        directory = '/Users/Admin/Dropbox/ML/MachineLearning_CS6140/CS6140_A_MacLeay/Homeworks'
        path = os.path.join(directory, 'hw4errors.pdf')
        print path
        plt.Errors([adaboost.local_errors]).plot_all_errors(path)
        roc = plt.ROC()
        roc.add_tpr_fpr_arrays(adaboost.tpr.values(), adaboost.fpr.values())
        roc.plot_ROC(os.path.join(directory, 'hw4_roc.pdf'))

def q2():
    """Boosting on UCI datasets"""
    pass

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
    pass

def q4():
    """
    ECOC
    """






