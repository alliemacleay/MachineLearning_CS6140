from sklearn.metrics import accuracy_score
from sklearn.neighbors import RadiusNeighborsClassifier, KernelDensity
from sklearn.grid_search import GridSearchCV
import CS6140_A_MacLeay.Homeworks.hw7 as hw7
import CS6140_A_MacLeay.Homeworks.HW7 as hw7u
import CS6140_A_MacLeay.Homeworks.HW3 as hw3u
import CS6140_A_MacLeay.Homeworks.HW4.data_load as dl
import CS6140_A_MacLeay.Homeworks.HW4 as hw4u
import CS6140_A_MacLeay.Homeworks.HW7.speedy as speedy
import numpy as np

__author__ = 'Allison MacLeay'

def tests_radius():
    i = 0
    j = 0
    k = 10
    X, y = testData()
    #print X
    X = np.concatenate([X, y.reshape((len(y), 1))], axis=1)
    X = [list(x.ravel()) for x in X]
    radius = [3, 5, 7]
    radius = [1e-1,1e-2,1e-3]  # for radius
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = speedy.Kernel(ktype=metric[j]).compute
    #ma = hw7u.Kernel(ktype=metric[j]).compute
    print 'spam radius is {}'.format(radius[i])
    clf = hw7u.MyKNN(radius=radius[i], metric=metric[j], outlier_label=-1)
    skclf = RadiusNeighborsClassifier(radius=radius[i], algorithm='brute', metric="euclidean", p=2, outlier_label=.5)
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
    print y_pred
    print 'SciKit Accuracy: {}  My Accuracy: {}'.format(accuracy_score(hw7.fix_y(y_test), hw7.fix_y(y_sci)), accuracy_score(hw7.fix_y(y_test), hw7.fix_y(y_pred)))


def tests_density():
    i = 0
    j = 2
    k = 10
    X, y = testData()
    print X
    X = np.concatenate([X, y.reshape((len(y), 1))], axis=1)
    X = [list(x.ravel()) for x in X]
    radius = [3, 5, 7]
    metric = ['minkowski', 'cosine', 'gaussian', 'poly2']
    ma = hw7u.Kernel(ktype=metric[j]).compute
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(X)
    clf = hw7u.MyKNN(metric=metric[j], density=True)

    bw = grid.best_estimator_.bandwidth
    print("best bandwidth: {0}".format(bw))

    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_
    skclf = KernelDensity(bandwidth=bw, kernel='gaussian')
    skclf.fit(X[:-10], y[:-10])
    print skclf.score_samples(X[-10:])
    return
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
    print y_pred
    print 'SciKit Accuracy: {}  My Accuracy: {}'.format(accuracy_score(hw7.fix_y(y_test), hw7.fix_y(y_sci)), accuracy_score(hw7.fix_y(y_test), hw7.fix_y(y_pred)))

def testGaussian2():
    all_X, all_y = testData()
    xX = np.concatenate([all_X, all_y.reshape((len(all_y), 1))], axis=1)
    xX = [list(x.ravel()) for x in xX]
    np.random.shuffle(xX)
    y = np.array(xX).T[2]
    X = np.array(xX).T[:2].T
    y_test, X_test = all_y[-10:], all_X[-10:]
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(all_X)
    bw = grid.best_estimator_.bandwidth
    print 'Bandwidth {}'.format(bw)


    myclassifier = hw7u.MyKNN(metric='gaussian', density=True, bandwidth=bw)
    print 'start MyKNN'
    myclassifier.fit(X, y)
    #print 'start scikit'
    #knnsci = skclassifier.fit(X, y)
    print 'start my pred'
    y_pred = myclassifier.predict(X_test)
    #print 'start sk pred'
    #y_sci = knnsci.score(X_test)
    #print 'SciKit Accuracy: {}  My Accuracy: {}'.format(accuracy_score(fix_y(y_test), fix_y(y_sci)), accuracy_score(fix_y(y_test), fix_y(y_pred)))
    print 'My Accuracy: {}'.format(accuracy_score(hw7.fix_y(y_test), hw7.fix_y(y_pred)))



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
    #tests_density()
    #testSpiralLoad()
    #testGaussian2()
    #testCython()
    #hw7.q1a()  # works
    #hw7.q1b()  # slow
    hw7.q2a()  # radius -- broken
    #hw7.q2b()  # density -- broken
    #hw7.q3a()  # dual perceptron -- broken
    #hw7.q3b()  # dual on spirals -- not done (contingent on dual perceptron)
    #hw7.q5()  # ridge feature selection -- confused
