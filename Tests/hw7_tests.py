from sklearn.metrics import accuracy_score
from sklearn.neighbors import RadiusNeighborsClassifier
import CS6140_A_MacLeay.Homeworks.hw7 as hw7
import CS6140_A_MacLeay.Homeworks.HW7 as hw7u
import CS6140_A_MacLeay.Homeworks.HW3 as hw3u
import CS6140_A_MacLeay.Homeworks.HW4.data_load as dl
import CS6140_A_MacLeay.Homeworks.HW4 as hw4u
import numpy as np

__author__ = 'Allison MacLeay'

def tests_radius():
    i = 0
    j = 0
    k = 10
    X, y = testData()
    X = np.concatenate([X, y.reshape((len(y), 1))], axis=1)
    X = [list(x.ravel()) for x in X]
    radius = [.5, .8, 3]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]).compute
    print 'spam radius is {}'.format(radius[i])
    clf = hw7u.MyKNN(radius=radius[i], metric=ma, density=True)
    skclf = RadiusNeighborsClassifier(radius=radius[i], algorithm='brute', metric=ma, p=2, outlier_label=.5)
    all_folds = hw3u.partition_folds(X, k)
    kf_train, kf_test = dl.get_train_and_test(all_folds, 0)
    y, X = hw4u.split_truth_from_data(kf_train)
    y_test, X_test = hw4u.split_truth_from_data(kf_test)
    print 'start scikit'
    knnsci = hw7u.KNN(classifier=skclf)
    print 'start MyKNN'
    knn = hw7u.KNN(classifier=clf)
    print 'start sk pred'
    y_sci = knnsci.predict(X_test, X, y)
    print 'start my pred'
    y_pred = knn.predict(X_test, X, y)
    print 'SciKit Accuracy: {}  My Accuracy: {}'.format(accuracy_score(hw7.fix_y(y_test), hw7.fix_y(y_sci)), accuracy_score(hw7.fix_y(y_test), hw7.fix_y(y_pred)))


def testData(y_ones=False):
    # Create the dataset
    rng = np.random.RandomState(1)
    X = rng.rand(100, 2)
    y = np.asarray([0] * 50 + [1] * 50)
    X[y == 1] += 2.0
    return X, y

def testSpiralLoad():
    X, y = dl.load_spirals()
    pass

def testCython():
    hw7u.testCython()


if __name__ == '__main__':
    #tests_radius()
    #testSpiralLoad()
    testCython()
    #hw7.q1a()
    #hw7.q1b()
    #hw7.q2a()
    #hw7.q2b()
    #hw7.q3a()
    #hw7.q3b()
    #hw7.q5()
