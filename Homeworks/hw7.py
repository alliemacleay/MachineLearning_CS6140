import CS6140_A_MacLeay.utils as utils
import CS6140_A_MacLeay.Homeworks.HW6 as hw6u
import CS6140_A_MacLeay.Homeworks.HW3 as hw3u
import CS6140_A_MacLeay.Homeworks.HW4.data_load as dl
import CS6140_A_MacLeay.Homeworks.HW4 as hw4u
import CS6140_A_MacLeay.Homeworks.HW7 as hw7u
#import CS6140_A_MacLeay.Homeworks.HW7.speedy as hw7u
import CS6140_A_MacLeay.utils.Perceptron as perc
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, KernelDensity
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier

__author__ = 'Allison MacLeay'

def q1a():
    """ KNN on spambase
    k = 1, j = 0
    SciKit Accuracy: 0.921908893709  My Accuracy: 0.921908893709
    k = 3, j = 0
    SciKit Accuracy: 0.919739696312  My Accuracy: 0.919739696312
    k = 7, j = 0
    SciKit Accuracy: 0.915401301518  My Accuracy: 0.915401301518
    """
    i = 0  # controls k
    j = 0  # controls the metric
    runSpamKNN(i, j, features='all')

def runSpamKNN(i, j, features='all'):
    n_neighbors = [1, 3, 7]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]).compute
    skclassifier = KNeighborsClassifier(n_neighbors=n_neighbors[i], algorithm='brute', metric=ma, p=2)
    myclassifier = hw7u.MyKNN(n_neighbors=n_neighbors[i], metric=ma)
    SpamClassifier(features, skclassifier, myclassifier)

def runSpamDensity(_i, j, features='all'):
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]).compute
    #skclassifier = KernelDensity(kernel=ma)
    skclassifier = KernelDensity()
    data = utils.pandas_to_data(utils.load_and_normalize_spam_data())
    k = 10
    myclassifier = hw7u.MyKNN(metric=ma, density=True)
    all_folds = hw3u.partition_folds(data, k)
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train)
    y_test, X_test = hw4u.split_truth_from_data(kf_test)
    print 'start MyKNN'
    knn = myclassifier.fit(X, y)
    print 'start scikit'
    knnsci = skclassifier.fit(X, y)
    print 'start my pred'
    y_pred = knn.predict(X_test, X, y)
    print 'start sk pred'
    y_sci = knnsci.score(X_test, X, y)
    print 'SciKit Accuracy: {}  My Accuracy: {}'.format(accuracy_score(fix_y(y_test), fix_y(y_sci)), accuracy_score(fix_y(y_test), fix_y(y_pred)))

def runSpamRadius(i, j, features='all'):
    radius = [.5, .8, 2.5]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]).compute
    print 'spam radius is {}'.format(radius[i])
    skclassifier = RadiusNeighborsClassifier(radius=radius[i], algorithm='brute', metric=ma, p=2, outlier_label=.5)
    myclassifier = hw7u.MyKNN(radius=radius[i], metric=ma, outlier_label=.5)
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
        k = 1, j = 2 (gaussian)
        k = 1, j = 3 (poly2)
    """
    i = 0  # controls k
    j = 1  # controls the metric
    n = 2000
    runDigitsKNN(i, j, n)

def runDigitsKNN(i, j, n):
    n_neighbors = [1, 3, 7]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]).compute
    skclf = KNeighborsClassifier(n_neighbors=n_neighbors[i], algorithm='brute', metric=ma, p=2)
    myclf = hw7u.MyKNN(n_neighbors=n_neighbors[i], metric=ma)
    runDigits(n, skclf, myclf)

def runDigitsDensity(_i, j, n):
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]).compute
    skclf = KernelDensity(algorithm='brute', metric=ma, p=2)
    myclf = hw7u.MyKNN(metric=ma, density=True)
    runDigits(n, skclf, myclf)

def runDigitsRadius(i, j, n):
    radius = [.5, .83, 1.3]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    print 'Digits radius is {}'.format(radius[i])
    ma = hw7u.Kernel(ktype=metric[j]).compute
    skclf = RadiusNeighborsClassifier(radius=radius[i], algorithm='brute', metric=ma, p=2, outlier_label=10)
    myclf = hw7u.MyKNN(radius=radius[i], metric=ma, outlier_label=10)
    runDigits(n, skclf, myclf)

def runDigits(n, skclf, myclf):
    mnsize = n
    df = hw6u.load_mnist_features(mnsize)
    data = utils.pandas_to_data(df)
    k = 10
    all_folds = hw3u.partition_folds(data, k)
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train, replace_zeros=False)
    y, X = np.asarray(y), np.asarray(X)
    y_test, X_test = hw4u.split_truth_from_data(kf_test, replace_zeros=False)
    y_test, X_test = np.asarray(y_test), np.asarray(X_test)
    print 'scikit fit'
    skclf = skclf.fit(X, y)
    print 'my fit'
    clf = OneVsRestClassifier(myclf).fit(X, y)
    print 'scikit predict'
    sk_pred = skclf.predict(X_test)
    print 'my predict'
    y_pred = clf.predict(X_test)
    print sk_pred
    print y_test
    print y_pred
    print 'SciKit Accuracy: {}  My Accuracy: {}'.format(accuracy_score(y_test, sk_pred), accuracy_score(y_test, y_pred))
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
   Digits + Cosine + R=0.83: test acc: 0.886

    Running Spam Radius
    spam radius is 0.8
    my predict euclidean
    2 outliers
    [42, 111]
    SciKit Accuracy: 0.60737527115  My Accuracy: 0.60737527115

    Running Digits Radius
    Loading 2000 records from haar dataset

    """
    print 'Running Spam Radius'
    # r = [1, 5, 10]
    runSpamRadius(2, 0)  # .833
    print 'Running Digits Radius'
    # radius, metric, n_records
    runDigitsRadius(1, 1, 2000)  # cosine r=.83 e=.886

def q2b():
    """
       B - KNN Kernel Density

       B) Spam + Gaussian(sigma=1.0): test acc: 0.910
   Digits + Guassian(sigma=1.0): test acc: 0.926
   Digits + Poly(degree=2): test acc: 0.550
    """
    runSpamDensity(0, 2)  # Gaussian  # expect .91
    runDigitsDensity(0, 2)  # Gaussian # expect .96
    runDigitsDensity(0, 3)  # Poly # expect .55



def q3a():
    """ A - Dual version on Perceptron
        B - Spirals
        Try to run the dual perceptron (with dot product) and conclude that the perceptron does not work.  Then run the dual perceptron with the Gaussian kernel and conclude the data is now separable.
        Expected accuracy dot product kernel : 50% average oscillating between 66% and 33%
        Expected accuracy with RBF kernel : 99.9% (depends on sigma)
    """
    homework_2_perceptron()

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
    i = 0  # controls k
    j = 0  # controls the metric
    runSpamKNN(i, j, features=top_five)

def relief(n):
    max_iters = 100
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
    while max_iters > loops:
        skclassifier = KNeighborsClassifier(n_neighbors=n_neighbors[i], weights=weights, algorithm='brute', metric=ma, p=2)
        print 'start scikit'
        knnsci = hw7u.KNN(classifier=skclassifier)
        #print 'start MyKNN'
        #knn = hw7u.KNN(classifier=myclassifier)
        print 'start sk pred'
        y_sci = knnsci.predict(X_test, X, y)
        #print 'start my pred'
        #y_pred = knn.predict(X_test, X, y)
        print 'SciKit Accuracy: {}  My Accuracy: {}'.format(accuracy_score(fix_y(y_test), fix_y(y_sci)), 'NA') #accuracy_score(fix_y(y_test), fix_y(y_pred)))
        loops += 1
        for j in range(len(X_test[0])): #feature
            closest_same = None
            closest_opp = None
            for i in range(len(X_test)):  # data

                for z_i in range(len(X)):
                    if y_sci[i] == y_test[i]:  # same
                        diff = X[z_i][j] - X_test[i][j] ** 2
                        if closest_same is None or diff < closest_same:
                            closest_same = diff
                    else:  # opp
                        if closest_opp is None or diff < closest_opp:
                            closest_opp = diff
            weights[j] -= (closest_same + closest_opp)
        print weights

    return sorted(zip(weights, range(len(weights))))[:n][1]





def fix_y(y):
    return [0 if y_i != -1 else y_i for y_i in y]






