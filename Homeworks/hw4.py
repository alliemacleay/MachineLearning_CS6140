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

def q1():
    spamData = hw3.pandas_to_data(hw3.load_and_normalize_spambase())
    adaboost = adab.AdaboostOptimal(7)
    adaboost.run(spamData)
    adaboost.print_stats()


