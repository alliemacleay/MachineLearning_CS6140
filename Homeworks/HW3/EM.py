import scipy
import CS6140_A_MacLeay.Homeworks.HW3 as hw3
import CS6140_A_MacLeay.utils as utils
from CS6140_A_MacLeay.utils.Stats import multivariate_normal
import numpy as np
import sys
import math

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
        self.probabilities = None  # m x n (univariate normal)
        self.likelihood = None  # R (multivariate normal)
        self.column_weights = None

    def set_mus(self, data):
        mu = []
        by_col = hw3.transpose_array(data) # to go by column
        for j in range(len(by_col)):
            mu.append(np.mean(by_col[j]))
        self.mu = mu

    def xrandom_mus(self, data):
        mu = []
        for j in range(len(data[0])):
            mu.append(data[np.ceil(np.random.random() * len(data))][j-1])
        self.mu = mu

class EMComp(object):
    def __init__(self):
        self.labels = None
        self.models = []  # k-sized array of EMModels
        self.llh = None
        self.k = 0

    def print_results(self, iteration=None):
        models = self.models
        if iteration is not None:
            print 'ITERATION {}'.format(iteration)
        for i in range(len(models)):
            print '\nmean_{}: {}'.format(i, models[i].mu)
            print 'cov_{}: {}'.format(i, models[i].sigma)
            print 'n{}: {}'.format(i, self.labels.count(i))
        print_weight = ''
        for i in range(len(models)):
            print_weight += '{} '.format(models[i].weight)
        print_weight += ' tot: {}'.format(sum([models[i].weight for i in range(len(models))] ) )
        print print_weight


    def emgm(self, data, k=2, max_iters=1, tol=1e-10):
        """
        data = mxn matrix
        max_iters = maximum iterations
        tol = convergence criteria
        """
        converged = False
        self.initialize(data, k)
        t=0
        last = None
        while not converged and t < max_iters:
            t += 1
            last = self.models[0].mu
            self.maximize(data)
            for ki in range(k):
                self.models[ki].likelihood = self.expectation(data, self.models[ki])
            self.print_results(iteration=t)
            converged = self.get_tol(tol, last)

    def get_tol(self, tolerance, last):
        if last is None:
            return True
        diff = np.array(last[0]) - self.models[0].mu
        delta = np.sqrt(np.dot(diff, diff.T))
        #print delta
        return tolerance > delta


    def initialize(self, data, k=2):
        # start with k = 2 and std_dev = 1
        self.k = k
        self.labels = [ki for ki in range(self.k)]
        models = [EMModel() for _ in range(self.k)]

        mucheat = mu_cheat(hw3.transpose_array(data), k)
        for ki in range(self.k):
            #models[ki].random_mus(data)
            models[ki].mu = mucheat[ki]

        self.labels = self.assign_labels(data, models)
        #self.labels = self.assign_labels2(data, model)

        self.prevent_empty(data)

        for ki in range(self.k):
            sub_data = hw3.get_sub_at_value(data, self.labels, ki)
            #models[ki].sigma = hw3.get_covar(sub_data)
            models[ki].sigma = hw3.get_covar(data)
            #models[ki].weight = float(len(sub_data)) / len(data)
            models[ki].weight = .5
            models[ki].likelihood = self.expectation(data, models[ki])  # multivarate_normal
        self.models = models

    def prevent_empty(self, data):
        classified = set(self.labels)
        while len(classified) < self.k:
            for i in range(self.k):
                if i not in classified:
                    #lonely_label = i
                    self.models[i].mu = self.models[i].random_mus(data)
                    self.labels = self.assign_labels2(data, self.models)
                    classified = set(self.labels)


    def expectation(self, data, model):
        """Compute probabilities
            store in self.model
        """
        mus = model.mu
        sigma = model.sigma
        weight = model.weight
        probabilities = []

        rho = np.zeros(len(data))   # rho is probabilities
        for r in range(len(data)):
            row = data[r]
            x_less = np.array(row) - np.array(mus)
            # the expectation <Zim> * P(x|Zim)

            rho[r] = multivariate_normal(sigma, x_less, .001) * weight
            #rho[r] = scipy.stats.multivariate_normal.pdf(data[r], mean=mus, cov=sigma) * weight
        #model.probabilities = hw3.univariate_normal(data, np.std(data), model.mu, model.weight)
        print 'RHO'
        print rho
        # log likelihood
        # llh = np.log(rho[i]/sum(rho))
        sum_rho = sum(rho)
        new_rho = []
        #return [float(rho[r])/sum_rho * weight for r in range(len(rho))]
        for r in range(len(rho)):
            new_rho.append(rho[r]/sum_rho) # * weight)
        return new_rho
        #return rho  # probabilities

    def maximize(self, data):
        """
        """
        for ki in range(self.k):
            mus = []
            model = self.models[ki]
            R = model.likelihood
            sum_row = 0.0
            for j in range(len(data)):
                sum_row += self.models[ki].likelihood[j] * data[j]

            mus = sum_row / sum(R)
            # New mus per column
            model.mu = mus
            self.models[ki] = model

        #self.labels = self.assign_labels(data, self.models)
        #self.labels = self.assign_labels2(data, self.models)
        #sigmas = self.get_sigmas(data, self.k)

        sum_weight = 0
        for ki in range(self.k):
            #sub_data = hw3.get_sub_at_value(data, self.labels, ki)

            #if(len(sub_data)) == 0:
            #    print 'ERROR empty subset'
            #self.models[ki].sigma = hw3.get_covar(sub_data)
            self.models[ki].sigma = self.get_sigma(data, ki)
            self.models[ki].weight = float(sum(self.models[ki].likelihood)) / len(data)
            sum_weight += self.models[ki].weight
            #self.models[ki].weight = 1

            #self.models[ki].weight = float(len(sub_data)) / len(data)

        for ki in range(self.k):
            self.models[ki].weight = self.models[ki].weight / sum_weight



    def get_sigma(self, data, ki):
        num = 0.0
        for j in range(len(data)):
            P = self.models[ki].likelihood[j]  # scalar
            diff = data[j] - self.models[ki].mu
            diffT = np.matrix(diff).T
            prod2 = np.dot(diffT, np.matrix(diff))
            num += P * prod2
            #print num
            #num += sum(self.models[ki].likelihood * prod)
        return num / sum(self.models[ki].likelihood)

    def assign_labels(self, data, models):
        labels = []
        for r in range(len(data)):
            min_distance = None
            row = data[r]
            for k in range(len(models)):
                distance = compute_distance(models[k].mu, row)
                if min_distance is None or distance < min_distance[0]:
                    min_distance = [distance, k]
            labels.append(min_distance[1])

        return labels

    def assign_labels2(self, data, models):
        labels = []
        max_likelihood = None
        for r in range(len(data)):
            for k in range(len(models)):
                R = models[k].likelihood
                if max_likelihood is None or R[r] > max_likelihood[0]:
                    max_likelihood = [R[r], k]
            labels.append(max_likelihood[1])
        return labels

    def test_convergence(self):
        converged = False
        llh = 0
        for ki in range(self.k):
            llh_v = np.log(self.models[ki].likelihood)
            llh += sum(llh_v)
        if self.llh is not None and llh - self.llh < self.tol * abs(llh):
            converged = True
            print 'CONVERGED!'
        self.llh = llh
        return converged



def compute_distance(mu, data):
    #TODO - find correct computation
    diff = mu - np.asarray(data)
    return np.dot(diff.T, diff)/len(data)

def xmu_cheat(by_col, k):
    mus = []
    for c in range(len(by_col)):
        minc = min(by_col[c]) #[0]
        maxc = max(by_col[c]) #[0]
        meanc = np.array(by_col[c]).mean()
        if k == 2:
            mus.append([float(meanc-minc)/2, float(maxc-meanc)/2])
        if k == 3:
            mus.append([minc, meanc, maxc])
    return mus

def xmu_cheat(data, k):
    mus = [[3,3],[7,4]]
    if k == 3:
        mus.append([5, 7])
    return mus

def mu_cheat(data, k):
    mus = [[0,0],[9,2]]
    if k == 3:
        mus.append([4, 9])
    return mus













