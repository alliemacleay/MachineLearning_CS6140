import pandas
from pandas.rpy.common import convert_to_r_dataframe
from rpy2 import robjects
from rpy2.robjects.lib import ggplot2
import numpy as np

__author__ = 'Allison MacLeay'

class Errors(object):
    def __init__(self, error_matrix):
        self.error_matrix = error_matrix
        self.colors = ['blue', 'red']

    @property
    def df(self):
        return pandas.DataFrame({'iteration': self.error_matrix[0].keys(), 'error': self.error_matrix[0].values()})

    def plot_all_errors(self, path):
        #print self.error_matrix[0]

        robjects.r['pdf'](path, width=14, height=8)

        #df = pandas.DataFrame({'iteration': range(1, len(self.error_matrix[0])), 'error': self.error_matrix[0]})
        #df = pandas.DataFrame({'fpr': range(10), 'tpr': range(10), 'theta': range(10)})
        df = self.df
        #print(df)
        gp = ggplot2.ggplot(convert_to_r_dataframe(df, strings_as_factors=True))
        gp += ggplot2.aes_string(x='iteration', y='error')
        gp += ggplot2.geom_line(color='blue')
        gp += ggplot2.geom_point(size=2)
        gp.plot()


class ROC(object):
    def __init__(self):
        self.auc = 0
        self.tpr = []
        self.fpr = []
        self.thetas = []

    def add_tp_tn(self, predict, truth, theta=None):
        tot_pos = sum(truth)
        tot_neg = len(truth) - sum(truth)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for p in range(len(predict)):
            val = predict[p]
            if val == 1:
                if val == truth[p]:
                    tp += 1
                else:
                    fp += 1
            else:
                if val == truth[p]:
                    tn += 1
                else:
                    fn += 1
        self.tpr.append(float(tp)/(tp + fn))
        self.fpr.append(float(fp)/(fp + tn))
        self.thetas.append(theta)

    def add_tpr_fpr_arrays(self, tpr, fpr, thetas=None):
        if thetas is None:
            thetas = np.ones(len(tpr))
        self.tpr = tpr
        self.fpr = fpr
        self.thetas = thetas

    def get_AUC(self):
        """(1/2) sumk=2 to m (xk - xk-1) (yk + yk-1).
        """
        df = self.df.sort('fpr')
        print(df)
        sumi = 0
        for i in range(1, len(self.fpr)):
            sumi += (df['fpr'].values[i] - df['fpr'].values[i -1]) * (df['tpr'].values[i] + df['tpr'].values[i - 1])
        self.auc = sumi * 1./2
        return self.auc

    def print_info(self):
        print '\n\n'
        for i in range(len(self.tpr)):
            if i < len(self.thetas) and self.thetas[i] is not None:
                str = 'theta: {:.4f}  '.format(self.thetas[i])
            else:
                str = '               '
            str += 'TPR: {:.4f}  FPR: {:.4f}'.format(self.tpr[i], self.fpr[i])
            print str
        print 'AUC: {:.4f}'.format(self.get_AUC())

    @property
    def df(self):
        return pandas.DataFrame({'fpr': self.fpr, 'tpr': self.tpr, 'theta': self.thetas})

    def plot_ROC(self, path):
        robjects.r['pdf'](path, width=14, height=8)

        df = self.df
        #print(df)
        gp = ggplot2.ggplot(convert_to_r_dataframe(df, strings_as_factors=True))
        gp += ggplot2.aes_string(x='fpr', y='tpr')
        gp += ggplot2.geom_line(color='blue')
        gp += ggplot2.geom_point(size=2)
        gp.plot()



