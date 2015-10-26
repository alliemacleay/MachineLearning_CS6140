from sklearn.metrics import roc_auc_score
from sklearn.metrics.scorer import roc_auc_scorer
from sklearn.mixture import GMM
from sklearn.naive_bayes import GaussianNB

__author__ = 'Allison MacLeay'

import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import CS6140_A_MacLeay.Homeworks.hw3 as hw3_run
import CS6140_A_MacLeay.utils.NaiveBayes as nb
import CS6140_A_MacLeay.Homeworks.HW3.EM as em
#import CS6140_A_MacLeay.Homeworks.HW3.EM2 as emv
import numpy as np



class Q1():
    def __init__(self):
        pass

    def test_q1(self):
        #testSpamDataLoad()
        #testPdToDict()
        #testTransposeArray()
        #test_get_mus()
        #test_covar_matrix()
        #test_GDA()
        #test_separate_x_and_y()
        #test_set_diag_min()
        test_q1()
        pass

def testSpamDataLoad():
    print type(hw3.load_and_normalize_spambase())

def testPdToDict():

    df = hw3.load_and_normalize_spambase()
    cols = df.columns[0:3]
    sub = utils.train_subset(df, cols, 5)
    print sub
    print hw3.pandas_to_data(sub)

def testTransposeArray():
    dfup = hw3.load_and_normalize_spambase()
    cols = dfup.columns[0:3]
    sub = utils.train_subset(dfup, cols, 5)
    up = hw3.pandas_to_data(sub)
    print up
    trans = hw3.transpose_array(up)
    print trans

def test_get_mus():
    arr = get_test_data()
    print arr
    print hw3.get_mus(arr)

def test_covar_matrix():
    arr = get_test_data()
    print arr
    y = [1, 0, 0, 0]
    print hw3.get_covar(arr, y)
    print np.cov(hw3.transpose_array(arr), y)

def test_set_diag_min():
    matrix = np.zeros(shape=(10, 10))
    print matrix
    hw3.set_diag_min(matrix, 5)
    print matrix

def test_GDA():
    arr = get_test_data()
    print arr
    covar = hw3.get_covar(arr, arr)
    print hw3.GDA(arr, hw3.get_mus(arr), covar)


""" run q1 """
def test_q1():
    hw3_run.q1()


""" get test data
"""
def get_test_data(num=4):
    arr = []
    for i in range(num):
        arr.append([1 + i, 2 + i, 3+i])
    return arr

def get_binary_test_data(num):
    arr = get_test_data(num)
    for r in range(len(arr)):
        arr[r].append(r % 2)
    return arr

def get_nb_data(var=0):
    arr = []
    arr.append([70 + get_rand(var), 4 + get_rand(var), 0])
    arr.append([23 + get_rand(var), 90 + get_rand(var), 1])
    arr.append([80 + get_rand(var), 1 + get_rand(var), 0])
    arr.append([4 + get_rand(var), 91 + get_rand(var), 1])
    arr.append([3 + get_rand(var), 14 + get_rand(var), 0])
    return arr

def get_rand(n):
    return n * np.random.random()

def get_nb_test_data(_):
    arr = []
    arr.append([9, 14])  # 0
    arr.append([82, 4])  # 0
    arr.append([69, 1])  # 0
    arr.append([7, 79])  # 1
    arr.append([23, 89])  # 1
    return arr

"""
Question 2
"""


class Q2():
    def test_q2(self):
        #test_partition_data()
        #test_get_data_and_mus()
        #test_NaiveBayes()
        #test_NaiveBayes_predict()
        #test_variance()
        #test_update_model()
        #test_update_model_change()
        #test_bayes()
        #test_agg_model()
        #test_agg_model2()
        #test_classify()
        test_q2_run()

    def test_plots(self):
        hw3_run.q2_plots()

def test_classify():
    row = [22,7,3,5,2,7,3,6,9,1,4,0]
    cutoffs = [0, 5, 7, 11]
    print hw3.classify(row, cutoffs)
    # return [6, 2, 3, 1]


def test_agg_model():
    agg = nb.NaiveBayes(0)
    m1 = [[0,0,0],[0,0,0],[0,0,0], 0]
    m2 = [[1,1,1],[1,1,1],[1,1,1], 1]
    agg.aggregate_model([m1, m2])
    print agg.model

def test_agg_model2():
    agg = nb.NaiveBayes(1)
    mod1 = nb.NaiveBayes(1)
    mod2 = nb.NaiveBayes(1)
    m1 = [{0:[0,0,0], 1:[0,0,0]}, 0]
    m2 = [{0:[1,1,1], 1:[1,1,1]}, 1]
    mod1.model = m1
    mod2.model = m2
    agg.aggregate_model2([mod1, mod2])
    print agg.model

def test_bayes():
    X, y = get_test_data_bayes()
    nb_model = nb.NaiveBayes(model_type, alpha=.001)
    nb_model.train(X, y)




def get_test_data_bayes():
    def gen_one(means):
        return np.random.normal(loc=means, scale=np.ones(len(means)),
                                size=(1000, len(means)))

    X_neg = gen_one([-10, -20])
    X_pos = gen_one([10, 20])
    X = np.concatenate([X_neg, X_pos], axis=0)
    y = np.asarray([0] * X_neg.shape[0] + [1] * X_pos.shape[0])
    return X, y

def test_partition_data():
    arr = get_test_data(303)
    print hw3.partition_folds(arr)

def test_get_data_and_mus():
    arr = get_test_data(4)
    truth_rows, data_rows, data_mus, y_mu = hw3_run.get_data_and_mus(arr)
    print 'unsplit {}'.format(arr)
    print 'truth rows {}'.format(truth_rows)
    print 'data rows {}'.format(data_rows)
    print 'data mus {}'.format(data_mus)
    print 'y_mu {}'.format(y_mu)

def test_NaiveBayes_predict():
    bayes = nb.NaiveBayes(2)
    arr = get_nb_data()
    test = get_nb_test_data(5)
    print arr
    truth_rows, data_rows, data_mus, y_mu = hw3.get_data_and_mus(arr)
    bayes.train(data_rows, truth_rows)
    print data_mus
    print bayes.model
    print bayes.predict(test)

def test_update_model():
    bayes = test_NaiveBayes()
    print bayes.model
    arr = get_nb_data()
    print arr
    truth_rows, data_rows, data_mus, y_mu = hw3_run.get_data_and_mus(arr)
    bayes.train(data_rows, truth_rows)
    print bayes.model

def test_update_model_change():
    bayes = test_NaiveBayes()
    print bayes.model
    arr = get_nb_data(5)
    print arr
    truth_rows, data_rows, data_mus, y_mu = hw3_run.get_data_and_mus(arr)
    bayes.train(data_rows, truth_rows)
    print bayes.model
    test = get_nb_test_data(5)
    print 'prediction'
    print bayes.predict(test)


def test_NaiveBayes():
    bayes = nb.NaiveBayes(2)
    arr = get_nb_data()
    print arr
    truth_rows, data_rows, data_mus, y_mu = hw3.get_data_and_mus(arr)
    bayes.train(data_rows, truth_rows)
    print bayes.model
    return bayes

def test_variance():
    a = get_test_data(8)
    arr = a[0]
    mu = utils.average(arr)
    sum = 0
    for i in range(len(arr)):
        sum += (arr[i] - mu)**2
    print float(sum/len(arr))

    print np.var(arr)

def test_separate_x_and_y():
    array = get_test_data(5, 3)
    print array
    print hw3.separate_X_and_y(array)


def get_test_data(n, m):
    array = []
    for r in range(n):
        row = []
        for col in range(m):
            row.append(np.random.random())
        array.append(row)
    return array

def test_q2_run():
    hw3_run.q2()

class Q4():
    def test_q4(self):
        #test_EMclass()
        #test_assign()
        #test_EM_validation()
        test_q4()


""" Question 4 #4 """
"""mean_1 [3,3]); cov_1 = [[1,0],[0,3]]; n1=2000 points
mean_2 =[7,4]; cov_2 = [[1,0.5],[0.5,1]]; ; n2=4000 points
You should obtain a result visually like this (you dont necessarily have to plot it)

B) Same problem for 2-dim data on file 3gaussian.txt , generated using a mixture of three Gaussians. Verify your  findings against the true parameters used generate the data below.
mean_1 = [3,3] ; cov_1 = [[1,0],[0,3]]; n1=2000
mean_2 = [7,4] ; cov_2 = [[1,0.5],[0.5,1]] ; n2=3000
mean_3 = [5,7] ; cov_3 = [[1,0.2],[0.2,1]]    ); n3=5000
"""
def test_q4():
    hw3_run.q4()
    pass

def test_assign():
    data = np.asarray(utils.load_gaussian(2))
    em_algo = em.EMComp()
    em_algo.initialize(data)
    print em_algo.models[0].mu
    em_algo.maximize(data)

def test_EM_validation():
    data = np.asarray(utils.load_gaussian(2))
    em_algo = emv.gaussianEM(data)
    #em_algo.initialize(data)
    #print em_algo.models[0].mu


def test_EMclass():
    for i in [2,3]:
        data = np.asarray(utils.load_gaussian(i))
        #em_algo = em.EMComp()
        #em_algo.initialize(data)
        #print em_algo.models[0].mu
        gmm = GMM(n_components=i, covariance_type='full').fit(data)
        print(gmm.means_)
        print(gmm.covars_)






def UnitTests():
    #Q1().test_q1()
    #Q2().test_q2()
    #Q2().test_plots()
    Q4().test_q4()



if __name__=='__main__':
    UnitTests()
    #pass
