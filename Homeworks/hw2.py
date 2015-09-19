import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import scipy as sp
import utils
import utils.Stats as mystats


__author__ = 'Admin'


def compute_cost(x, y, delta):
    """compute cost (J)"""
    m = y.size
    predicted = x.dot(delta)   # matrix multiplication
    errors = predicted - y
    J = (1.0 / (2 * m)) * errors.T.dot(errors)  # squared errors
    return J

def gradient_descent(df, y_param, iterations, delta):
    """ multivariate gradient descent function """
    y = df[y_param].as_matrix()
    xcols = df.columns.tolist()
    xcols.remove(y_param)
    X = df[xcols].as_matrix()
    errors = []
    m = y.size
    columns = len(X[1, :])
    print columns
    theta = np.zeros(shape=(columns, 1))
    for i in range(0, iterations):
        # for each iteration
        predictions = X.dot(theta)  # matrix multiplication theta * X
        for col in range(0, columns):  # for each column as a feature
            temp = X[:, col]  # create a new matrix with just this column
            temp.shape = (m, 1)
            errors_x = (predictions - y) * temp  # (expected - observed) * theta
            theta[col][0] = theta[col][0] - delta * (1.0/m) * errors_x.sum()
            #  New theta values computed from the last theta values minus (learning parameter * average error)
        errors.append(compute_cost(X, y, theta))  # Keep track of error history to validate that it gets snaller each time
    return theta, errors


def logistic_regression(dftrain, dftest, predict_col):
    """ Logistic Regression for HW2 part B"""
    features = dftrain.columns.tolist()
    features.remove(predict_col)
    cls = LogisticRegression()
    cls.fit(dftrain[features], dftrain[predict_col])
    predictions = cls.predict(dftest[features])
    print predictions
    print mystats.compute_ACC(predictions, dftest[predict_col])


def do2A():
    """
    HW 2A
    Train linear regression using gradient descent on spambase and housing data
    """
    print('HW2 A. Gradient descent with housing and spam data sets')
    num_iters = 50
    learning_param = 0.25
    housingData_test, housingData_train = utils.load_and_normalize_housing_set()
    theta, error_matrix = gradient_descent(housingData_test, 'MEDV', num_iters, learning_param)
    print('Errors for housing set')
    print error_matrix
    print('theta for housing set')
    print theta

def do2B():
    hd_test, hd_train = utils.load_and_normalize_housing_set()
    logistic_regression(hd_train, hd_test, 'MEDV')


def homework2():
    do2A()
    do2B()