import numpy as np
import pandas as pd

import os
from abc import ABCMeta, abstractmethod

PATH = os.path.dirname(__file__) + '/'


class DataSet:
    __metaclass__ = ABCMeta

    x = None
    y = None

    file_path = ""

    @abstractmethod
    def load(self):
        print 'x.shape:', self.x.shape
        print 'y.shape:', self.y.shape, '\n'
        print 'y unique:', np.unique(self.y).tolist()

    def split(self, p_test):
        if not 0 < p_test < 1:
            raise NameError('Wrong split')

        border = int((1.0 - p_test) * len(self.x))
        perm = np.random.permutation(len(self.x))
        train_index, test_index = perm[:border], perm[border:]

        X_train, X_test = self.x[train_index], self.x[test_index]
        Y_train, Y_test = self.y[train_index], self.y[test_index]

        print 'Y_train: ', np.unique(Y_train)
        print 'Y_test:  ', np.unique(Y_train)

        return X_train, Y_train, X_test, Y_test


class IrisDataSet(DataSet):
    def __init__(self):
        self.file_path = PATH + "iris.data"

    def load(self):
        data = pd.read_csv(self.file_path, sep=',', header=None)
        data[4] = data[4].astype('category')
        data[4] = data[4].cat.rename_categories([0, 1, 2])
        data = data.values.astype(float)

        self.x, self.y = data[:, :-1], data[:, -1]
        super(IrisDataSet, self).load()
        return self


class WineDataSet(DataSet):
    def __init__(self):
        self.file_path = PATH + "wine.data"

    def load(self):
        data = pd.read_csv(self.file_path, sep=',', header=None)
        data = data.values.astype(float)
        self.x, self.y = data[:, 1:], data[:, 0]
        super(WineDataSet, self).load()
        return self


class BupaDataSet(DataSet):
    def __init__(self):
        self.file_path = PATH + "bupa.data"

    def load(self):
        data = pd.read_csv(self.file_path, sep=',', header=None)
        data = data.values.astype(float)
        self.x, self.y = data[:, :-1], data[:, -1]
        super(BupaDataSet, self).load()
        return self


if __name__ == "__main__":
    pass