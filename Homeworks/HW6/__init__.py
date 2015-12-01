import pandas as pd
import numpy as np


__author__ = 'Allison MacLeay'

def load_mnist_features(n):
    csv_file = 'df_save_img_everything.csv'
    print 'Loading {} records from haar dataset'.format(n)
    df = pd.read_csv(csv_file)
    del df['Unnamed: 0']

    skr = set(np.random.choice(range(len(df)), size=n, replace=False))  # might not be unique
    df = df[[idx in skr for idx in xrange(len(df))]]
    return df



