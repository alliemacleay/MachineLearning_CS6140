import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import numpy as np

__author__ = 'Allison MacLeay'

"""EM algorithm
E-step: compute all expectations (k) to fill in Y according
to current parameters (theta)
- for all examples j and for all values k for Yj
    compute P(Yj=k|xj,theta)

M-step: Re-estimate the parameters with weighted MLE estimates
- set theta = argmax SUMj SUMk P(Yj=k|xj, theta)log(P(Yj=k|xj, theta)

E and M have closed form solutions

start with 2 groups (k=2)
start with 1 std_dev: np.std(data)
estimate mu1 and mu2

:: Notation.  lamda(t) = {[mu1(t), mu2(t), ... muk(t)}
    t = # of iterations over EM algorithm - add a new mu each iteration
    !!! doesn't look like we add a new mu...

E step figures out which class to put the data point in
M step figures out the probability that it is in the expected class

"""

class EMModel(object):
    def __init__(self):
        self.mu = None  # vector of mus
        self.sigma = None
        self.weight = None

    def set_mus(self, data):
        mu = []
        by_col = hw3.transpose_array(data) # to go by column
        for j in range(len(by_col)):
            mu.append(np.mean(by_col[j]))
        self.mu = mu

class EMComp(object):
    def __init__(self):
        self.labels = None
        self.models = []  # k-sized array of EMModels
        self.llh = None
        self.k = 0

    def emgm(self):
        pass

    def initialize(self, data):
        # start with k = 2 and std_dev = 1
        self.k = 2
        model = [EMModel() for _ in range(self.k)]
        for ki in range(self.k):
            model[ki].set_mus(data)
            model[ki].sigma = 1
        self.models = model


    def expectation(self):
        """Compute probabilities
            store in self.model
        """
        pass

    def maximize(self):
        """
        """
        model = ''
        return model

    def loggausspdf(self, data):
        d = len(data)

