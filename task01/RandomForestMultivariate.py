import numpy as np

from itertools import izip

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.metrics import log_loss


class Node:
    nodes = [None, None]
    cls = None
    prediction = None

    def __init__(self, depth, min_impurity_split=1e-7):
        self.min_impurity_split = min_impurity_split
        self.depth = depth

    def fit(self, x, y):
        if np.unique(y).shape[0] == 1:
            self.prediction = y[0]
        else:
            self.cls = LogisticRegression(C=1e7, tol=1e-7)
            self.cls.fit(x, y)
            y_pred = self.cls.predict(x)

            if sum(y_pred == 0) > 0 and sum(y_pred == 1) > 0:
                self.nodes = [Node(self.depth + 1), Node(self.depth + 1)]

                mask = y_pred == 0
                self.nodes[0].fit(x[mask], y[mask])

                mask = y_pred == 1
                self.nodes[1].fit(x[mask], y[mask])
            else:
                self.prediction = int(sum(y_pred == 1) > 0)

    def check_nodes_exist(self):
        return all(node is not None for node in self.nodes)

    def predict_proba(self, x):
        if self.check_nodes_exist():
            result = np.zeros(shape=len(x))

            y_pred = self.cls.predict(x)
            mask = y_pred == 0
            if sum(mask) > 0:
                result[mask] = self.nodes[0].predict_proba(x[mask])

            mask = y_pred == 1
            if sum(mask) > 0:
                result[mask] = self.nodes[1].predict_proba(x[mask])

            return result
        else:
            return np.ones(shape=len(x)) if self.prediction else np.zeros(shape=len(x))


class DecisionTree:
    def __init__(self, min_impurity_split=1e-7):
        self.min_impurity = min_impurity_split
        self.root = None

    def fit(self, x, y):
        self.root = Node(1, self.min_impurity)
        self.root.fit(x, y)

    def get_depth(self):
        queue = [self.root]
        depth = 0
        while len(queue) > 0:
            el = queue[0]
            depth = max(depth, el.depth)
            queue.extend([el_ for el_ in el.nodes if el_ is not None])
            queue = queue[1:]
        return depth

    def predict_proba(self, x):
        pred = self.root.predict_proba(x)
        return np.vstack((1.0 - pred, pred)).T


class RandomForest:
    cl_forest = []
    cl_index_f = []

    def __init__(self, n_trees=100, p_items=0.8, p_features=0.8, min_impurity_split=1e-7):
        self.n_trees = n_trees
        self.p_items = p_items
        self.p_features = p_features
        self.min_impurity_split = min_impurity_split

    def fit(self, x, y):
        self.cl_forest = []
        self.cl_index_f = []

        for tree in range(self.n_trees):
            index_i = np.random.choice(x.shape[0], size=int(self.p_items * x.shape[0]))
            index_f = np.random.choice(x.shape[1], size=int(self.p_features * x.shape[1]), replace=False)
            x_train, y_train = x[index_i, ], y[index_i]
            x_train = x_train[:, index_f]

            cl = DecisionTree(self.min_impurity_split)
            cl.fit(x_train, y_train)
            self.cl_forest.append(cl)
            self.cl_index_f.append(index_f)

            if (tree + 1) % 10 == 0:
                print '\r{} of {} iters have passed...'.format(tree + 1, self.n_trees),

        print '\rFit is complete!'

    def predict_proba(self, x):
        pred = [cl.predict_proba(x[:, index_f]) for cl, index_f in izip(self.cl_forest, self.cl_index_f)]
        return np.asarray(pred).mean(axis=0)

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)


def choose_class(y, k):
    return (y == k).astype(int)


if __name__ == '__main__':
    iris = datasets.load_iris()
    kf = KFold(n_splits=10, shuffle=True)
    scores = [{'train': [], 'test': []} for k in range(3)]

    for train_index, test_index in kf.split(iris.data):
        perm = np.random.permutation(150)

        X_train, X_test = iris.data[train_index], iris.data[test_index]
        Y_train, Y_test = iris.target[train_index], iris.target[test_index]

        for k in range(3):
            cl = RandomForest(n_trees=100, p_items=1.0, p_features=0.8, min_impurity_split=0)
            cl.fit(X_train, choose_class(Y_train, k))

            Y_pred = cl.predict_proba(X_train)
            scores[k]['train'] = log_loss(choose_class(Y_train, k), Y_pred)
            Y_pred = cl.predict_proba(X_test)
            scores[k]['test']  = log_loss(choose_class(Y_test, k), Y_pred)

    for k in range(3):
        for c in ['train', 'test']:
            print "k = {}. {:5} logloss = {:.4f}".format(k, c, np.mean(scores[k][c]))
