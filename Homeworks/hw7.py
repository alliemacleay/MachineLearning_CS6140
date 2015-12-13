import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.Homeworks.HW6 as hw6u
import CS6140_A_MacLeay.Homeworks.HW3 as hw3u
import CS6140_A_MacLeay.Homeworks.HW4.data_load as dl
import CS6140_A_MacLeay.Homeworks.HW4 as hw4u
import CS6140_A_MacLeay.Homeworks.HW7 as hw7u
#import CS6140_A_MacLeay.Homeworks.HW7.speedy as hw7u
import CS6140_A_MacLeay.utils.Perceptron as perc
import CS6140_A_MacLeay.Homeworks.HW7.DualPerceptron as dperc
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, KernelDensity
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import sklearn.linear_model as lm
from mlpy import *

__author__ = 'Allison MacLeay'

def q1a():
    """ KNN on spambase
    k = 1, j = 0
    SciKit Accuracy: 0.921908893709  My Accuracy: 0.921908893709
    k = 2, j = 0
    SciKit Accuracy: 0.908893709328  My Accuracy: 0.908893709328
    k = 3, j = 0
    SciKit Accuracy: 0.919739696312  My Accuracy: 0.919739696312
    k = 7, j = 0
    SciKit Accuracy: 0.915401301518  My Accuracy: 0.915401301518
    """
    i = 0  # controls k
    j = 0  # controls the metric
    runSpamKNN(i, j, features='all')

def runSpamKNN(i, j, features='all'):
    n_neighbors = [1, 3, 7, 2]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]).compute
    skclassifier = KNeighborsClassifier(n_neighbors=n_neighbors[i], algorithm='brute', metric=ma, p=2)
    myclassifier = hw7u.MyKNN(n_neighbors=n_neighbors[i], metric='euclidean')
    SpamClassifier(features, skclassifier, myclassifier)

def runSpamDensity(_i, j, features='all'):
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    data = utils.pandas_to_data(utils.load_and_normalize_spam_data())
    k = 10
    all_folds = hw3u.partition_folds(data, k)
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train)
    y_test, X_test = hw4u.split_truth_from_data(kf_test)

    print(len(X))
    print(len(X_test))

    myclassifier = hw7u.MyKNN(metric='cosine_similarity', density=True)
    print 'start MyKNN'
    myclassifier.fit(X, y)
    #print 'start scikit'
    #knnsci = skclassifier.fit(X, y)
    print 'start my pred'
    y_pred = myclassifier.predict(X_test)
    print(y_pred)
    #print 'start sk pred'
    #y_sci = knnsci.score(X_test)
    #print 'SciKit Accuracy: {}  My Accuracy: {}'.format(accuracy_score(fix_y(y_test), fix_y(y_sci)), accuracy_score(fix_y(y_test), fix_y(y_pred)))
    print '2b: My Accuracy: {}'.format(accuracy_score(fix_y(y_test), fix_y(y_pred)))

def runSpamRadius(i, j, features='all'):
    radius = [.5, .8, 2.5]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]).compute
    print 'spam radius is {} distance metric is {}'.format(radius[i], metric[j])
    #skclassifier = RadiusNeighborsClassifier(radius=radius[i], algorithm='brute', metric='euclidean', p=2, outlier_label=-1)
    skclassifier = RadiusNeighborsClassifier(radius=radius[i], algorithm='brute', metric='euclidean', outlier_label=-1)
    myclassifier = hw7u.MyKNN(radius=radius[i], metric=metric[j], outlier_label=-1)
    SpamClassifier(features, skclassifier, myclassifier)



def SpamClassifier(features, skclassifier, myclassifier):
    data = utils.pandas_to_data(utils.load_and_normalize_spam_data())
    k = 10
    if features != 'all':
        # Only use the features passed in the features array
        new = []
        t = utils.transpose_array(data)
        for i in xrange(len(t)):
            if i in features:
                new.append(t[i])
            data = utils.transpose_array(t)
    all_folds = hw3u.partition_folds(data, k)
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train)
    y_test, X_test = hw4u.split_truth_from_data(kf_test)
    print 'start MyKNN'
    knn = hw7u.KNN(classifier=myclassifier)
    print 'start scikit'
    knnsci = hw7u.KNN(classifier=skclassifier)
    print 'start my pred'
    y_pred = knn.predict(X_test, X, y)
    print 'My Accuracy: {}'.format(accuracy_score(fix_y(y_test), fix_y(y_pred)))
    print 'start sk pred'
    y_sci = knnsci.predict(X_test, X, y)
    print 'SciKit Accuracy: {}  My Accuracy: {}'.format(accuracy_score(fix_y(y_test), fix_y(y_sci)), accuracy_score(fix_y(y_test), fix_y(y_pred)))




def q1b():
    """ KNN on digits
        k = 1, j = 0, n = 2000
        SciKit Accuracy: 0.85  My Accuracy: 0.85
        k = 3, j = 0
        k = 7, j = 0
        k = 1, j = 1 (cosine)
        n:2000 SciKit Accuracy: 0.895  My Accuracy: 0.895
        k = 3, j = 1
        k = 7, j = 1
        k = 1, j = 2 (gaussian)
        k = 3, j = 2 (gaussian)

        k = 7, j = 2 (gaussian)
        k = 1, j = 3 (poly2)
        k = 3, j = 3 (poly2)
        k = 7, j = 3 (poly2)
    """
    i = 1  # controls k [1,3,7,2]
    j = 2  # controls the metric
    n = 2000
    runDigitsKNN(i, j, n)

def runDigitsKNN(i, j, n):
    n_neighbors = [1, 3, 7]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]+'_sci').compute
    skclf = KNeighborsClassifier(n_neighbors=n_neighbors[i], algorithm='brute', metric=ma, p=2)
    myclf = hw7u.MyKNN(n_neighbors=n_neighbors[i], metric=metric[j])
    runDigits(n, skclf, myclf)

def runDigitsDensity(_i, j):
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]+'_sci').compute
    skclf = KernelDensity(metric=ma)
    myclf = hw7u.MyKNN(metric=metric[j], density=True)
    runDigits(n, skclf, myclf)

def runDigitsRadius(i, j, n):
    radius = [.5, .1, 1.3]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    print 'Digits radius is {} metric is {}'.format(radius[i], metric[j])
    ma = hw7u.Kernel(ktype=metric[j]+'_sci').compute
    #ma = cosine_distances
    skclf = RadiusNeighborsClassifier(radius=radius[i], algorithm='brute', metric=ma, outlier_label=9)
    myclf = hw7u.MyKNN(radius=radius[i], metric='cosine', outlier_label=-1)
    runDigits(n, skclf, myclf)

def runDigits(n, skclf, myclf):
    mnsize = n
    df = hw6u.load_mnist_features(mnsize)
    data = utils.pandas_to_data(df)
    k = 10
    all_folds = hw3u.partition_folds(data, k)
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train, replace_zeros=False)
    y, X = np.asarray(y, dtype=np.float), np.asarray(X)
    y_test, X_test = hw4u.split_truth_from_data(kf_test, replace_zeros=False)
    y_test, X_test = np.asarray(y_test), np.asarray(X_test, dtype=np.float)
    print 'my fit'
    clf = OneVsRestClassifier(myclf).fit(X, y)
    print 'scikit fit'
    skclf = skclf.fit(X, y)
    print 'my predict'
    y_pred = clf.predict(X_test)
    myacc = accuracy_score(y_test, y_pred)
    print '({})'.format(myacc)
    print 'scikit predict'
    sk_pred = skclf.predict(X_test)
    print sk_pred
    print y_test
    print y_pred
    print 'SciKit Accuracy: {}  My Accuracy: {}'.format(accuracy_score(y_test, sk_pred), myacc)
    #print 'My Accuracy: {}'.format(accuracy_score(y_test, y_pred))


def q2a():
    """A - KNN fixed window

    Spam
    max euclid = 5.0821092096899152
    min euclid = 3.8265101632475996e-08

    Digits
    max_euclid = 7.862600580777185
    min_euclid = 0.0041151139794844242
    mean_euclid = 1.2903757736212245

    A) Spam + Euclidian + R=2.5: test acc: 0.833
    SciKit Accuracy: 0.605206073753  My Accuracy: 0.605206073753
   Digits + Cosine + R=0.83: test acc: 0.886
   n = 2000 SciKit Accuracy: 0.18  My Accuracy: 0.175  12/13 14:40

   Digits + Cosine + R=0.1:
   n = 2000 SciKit Accuracy: 0.84  My Accuracy: 0.815


    Running Spam Radius
    spam radius is 0.8
    my predict euclidean
    2 outliers
    [42, 111]
    SciKit Accuracy: 0.60737527115  My Accuracy: 0.60737527115

    Running Digits Radius
    Loading 2000 records from haar dataset
    radius = [.5, .8, 2.5]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']

    """
    print 'Running Spam Radius'
    # r = [1, 5, 10]
    # (radius, metric)
    #runSpamRadius(2, 0)  # runSpamRadius(2, 0)  e=.833
    print 'Running Digits Radius'
    # radius, metric, n_records
    runDigitsRadius(1, 1, 2000)  # cosine r=.83 e=.886

def q2b():
    """
       B - KNN Kernel Density

       B) Spam + Gaussian(sigma=1.0): test acc: 0.910
   Digits + Guassian(sigma=1.0): test acc: 0.926
   Digits + Poly(degree=2): test acc: 0.550

   Cosine Similarity - My Accuracy: 0.8568329718
    """
    runSpamDensity(0, 2)  # Gaussian  # expect .91
    #runDigitsDensity(0, 2)  # Gaussian # expect .96
    #runDigitsDensity(0, 3)  # Poly # expect .55



def q3a():
    """ A - Dual version on Perceptron
        B - Spirals
        Try to run the dual perceptron (with dot product) and conclude that the perceptron does not work.  Then run the dual perceptron with the Gaussian kernel and conclude the data is now separable.
        Expected accuracy dot product kernel : 50% average oscillating between 66% and 33%
        Expected accuracy with RBF kernel : 99.9% (depends on sigma)
    """
    np.random.seed(2)
    test, train = utils.load_perceptron_data()
    c = train.columns[:-1]
    y_train = list(train[4])
    X_train = train[c].as_matrix()
    y_test = list(test[4])
    X_test = test[c].as_matrix()

    dual_perc = dperc.DualPerceptron(T=10)
    dual_perc.fit(X_train, y_train)
    y_pred = dual_perc.predict(X_test)
    print '3a: Dual Percepton AUC: {}'.format(roc_auc_score(y_test, dual_perc.decision_function(X_test)))
    print '3a: Dual Percepton Accuracy: {}'.format(accuracy_score(y_test, y_pred))

def q3b():
    X, y = dl.load_spirals()

def homework_2_perceptron():
    """ Perceptron """
    test, train = utils.load_perceptron_data()
    print test[4]
    print train.head(5)
    model = perc.Perceptron(train, 4, .05, 100)


def q5():
    """ RELIEF algorithm - Feature selection with KNN """
    top_five = relief(5)
    print top_five
    i = 0  # controls k
    j = 0  # controls the metric
    runSpamKNN(i, j, features=top_five)

def relief(n):
    max_iters = 1
    j = 0
    i = 1
    n_neighbors = [1, 3, 7]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]).compute
    data = utils.pandas_to_data(utils.load_and_normalize_spam_data())
    k = 10
    all_folds = hw3u.partition_folds(data, k)
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train)
    y_test, X_test = hw4u.split_truth_from_data(kf_test)
    loops = 0
    weights = np.zeros(len(X[0]))
    loops += 1
    for j in range(len(X[0])): #feature

        for i in range(len(X)):  # data
            closest_same = None
            closest_opp = None
            for z_i in range(len(X)):
                if z_i == i:
                    continue
                diff = (X[z_i][j] - X[i][j]) ** 2
                if y[z_i] == y[i]:  # same
                    if closest_same is None or diff < closest_same:
                        closest_same = diff
                else:  # opp
                    if closest_opp is None or diff < closest_opp:
                        closest_opp = diff
            weights[j] += (-closest_same + closest_opp)
    print weights

    return sorted(zip(weights, range(len(weights))), reverse=True)[:n][1]





def fix_y(y):
    return [0 if y_i != -1 else y_i for y_i in y]






