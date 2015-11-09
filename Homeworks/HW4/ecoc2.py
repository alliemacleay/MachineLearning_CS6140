import re
import numpy as np
from sklearn.linear_model import LogisticRegression
import CS6140_A_MacLeay.utils.Adaboost as adar
import CS6140_A_MacLeay.utils.Adaboost_compare as adac
import pandas as pd
import os
from sklearn.feature_selection import SelectKBest

__author__ = ''


def get_bits(val, length):
    """\
    Gets an array of bits for the given integer (most significant digits first),
    padded with 0s up to the desired length

    """
    bits = [int(bit_val) for bit_val in '{:b}'.format(val)]
    padding = [0] * max(0, length - len(bits))
    return padding + bits


class ECOCClassifier(object):
    """Implements multiclass prediction for any binary learner using Error Correcting Codes"""
    def __init__(self, learner=adar.AdaboostRandom, verbose=False, encoding_type='exhaustive'):
        #def __init__(self, learner=adac.AdaboostOptimal, verbose=False, encoding_type='exhaustive'):
        """\
        :param learner: binary learner to use
        """
        self.learner = learner
        self.verbose = verbose
        assert encoding_type in {'exhaustive', 'one_vs_all'}
        self.encoding_type = encoding_type

        # Things we'll estimate from data
        self.ordered_y = None
        self.encoding_table = None
        self.classifiers = None

    def _create_one_vs_all_encoding(self, ordered_y):
        """\
        Creates an identity encoding table. Much faster than exhaustive but gives suboptimal results.
        Useful for debugging.

        """
        return np.identity(len(ordered_y))

    def _create_exhaustive_encoding(self, ordered_y):
        """\
        Creates an exhaustive encoding table for the given set of unique label values

        :param ordered_y: unique labels present in the data, in a fixed order
        :return: matrix of size #unique labels x #ECOC functions

        """
        # Given k labels, the maximum number of unique binary encodings is 2 ** k. Of those, two are unary
        # (all 0s and all 1s), and as such not useful. This gives a total number of useful encodings equal to
        # 2 ** k - 2. Now we note that inverted encodings are equivalent, which means that the practical
        # number of unique, non-redundant encodings is only (2 ** k - 2)/2 = 2 ** (k - 1) - 1.
        n_functions = 2 ** (len(ordered_y) - 1) - 1

        # We generate the signature for each function by enumerating binary numbers between 1 and 2 ** k - 1,
        # making sure we don't include inverses.
        encodings = []
        for j in xrange(2 ** len(ordered_y) - 2):
            enc = tuple(get_bits(j + 1, len(ordered_y)))
            inv_enc = tuple([1 - x for x in enc])
            if enc not in encodings and inv_enc not in encodings:
                encodings.append(enc)

        encoding_table = np.array(encodings).T
        assert encoding_table.shape[1] == n_functions

        # Sanity check 1: make sure all functions have non-trivial encodings (with both 0s and 1s)
        for j in xrange(encoding_table.shape[1]):
            if len(set(encoding_table[:, j])) < 2:
                raise ValueError('Bad encoding. Function {} is unary.'.format(j))

        # Sanity check 2: make sure all encodings are unique
        encodings = [tuple(encoding_table[:, j]) for j in xrange(encoding_table.shape[1])]
        if len(encodings) != len(set(encodings)):
            raise ValueError('Some encodings are duplicated')

        if self.verbose:
            print('Encoding OK')

        return encoding_table

    def _encode_y(self, y, function_idx):
        """\
        Binarizes a multi-class vector y using the given function.

        :param y: multi-class label vector
        :param function_idx: which function to use for the encoding (between 0 and self.encoding_table.shape[1] - 1)
        :return: label vector binarized into 0s and 1

        """
        def encode_one(y_val):
            """\
            Encodes a single multi-class label.

            :param y_val: single label value
            :return: y_val encoded as either 0 or 1

            """
            y_idx = self.ordered_y.index(y_val)
            return self.encoding_table[y_idx, function_idx]

        # Check that the requested function is valid
        assert 0 <= function_idx < self.encoding_table.shape[1]

        # Binarize using the function's encoding
        return np.asarray([encode_one(y_val) for y_val in y])

    def fit(self, X, y):
        """Fits the classifier on data"""
        self.ordered_y = sorted(set(y))

        if self.encoding_type == 'exhaustive':
            self.encoding_table = self._create_exhaustive_encoding(self.ordered_y)
        else:
            self.encoding_table = self._create_one_vs_all_encoding(self.ordered_y)

        self.classifiers = []

        for function_idx in xrange(self.encoding_table.shape[1]):
            if self.verbose:
                print('Fit function {}/{}'.format(function_idx + 1, self.encoding_table.shape[1]))

            encoded_y = self._encode_y(y, function_idx)
            self.classifiers.append(self.learner().fit(X, encoded_y))

        return self

    def predict(self, X):
        """Predicts crisp labels for samples in a matrix"""
        def predict_one(idx, signature):
            """Predicts the label for a single sample"""
            # Compute hamming distance between our prediction and each label's encoding
            hamming_dist = {}
            for y_val, row in zip(self.ordered_y, self.encoding_table):
                hamming_dist[y_val] = np.sum(np.abs(signature - row))

            # Pick the label with the minimum hamming distance
            return min(hamming_dist.keys(), key=lambda y_val: hamming_dist[y_val])

        # Matrix of n_samples x n_functions. Each row is a vector of labels from each of the classifiers.
        signatures = np.array([cls.predict(X) for cls in self.classifiers]).T

        return np.asarray([predict_one(idx, sig) for idx, sig in enumerate(signatures)])

    @property
    def encoding_as_dataframe(self):
        """Returns the encoding table as a pandas DataFrame. Useful for debugging"""
        bit_columns = [str(x) for x in xrange(self.encoding_table.shape[1])]
        df = pd.DataFrame(data=self.encoding_table, columns=bit_columns)
        df['y'] = self.ordered_y
        return pd.DataFrame(df, columns=['y'] + bit_columns)

q4_slct = None
def cached(func):
    def inner(*args, **kwargs):
        X, y = func(*args, **kwargs)
        global q4_slct
        if q4_slct is None:
            q4_slct = SelectKBest(k=200).fit(X, y)
        X = q4_slct.transform(X)
        return X, y
    return inner

@cached
def parse_8newsgroup(path):
    """Parses 8newsgroup data from a directory, returning X and y"""
    mat_path = os.path.join(path, "feature_matrix.txt")
    feat_path = os.path.join(path, "feature_settings.txt")

    with open(feat_path, 'r') as f:
        n_features = sum([1 for _ in f])

    y = []
    rows = []
    with open(mat_path, 'r') as f:
        i = 0
        for line in f:

            toks = re.split(r'\s+', line.strip())
            label = int(toks[0])
            sparse_features = {int(feat): float(val) for feat, val in [tok.split(':') for tok in toks[1:]]}

            # Sparse -> dense
            row_data = np.zeros(n_features)
            for feat_idx, val in sparse_features.iteritems():
                row_data[feat_idx] = val

            y.append(label)
            rows.append(row_data)
    X = np.array(rows)
    return X, y




