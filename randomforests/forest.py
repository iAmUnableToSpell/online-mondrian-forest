from .tree_node import MondrianTreeClassifier
import numpy as np

class MondrianForestClassifier(object):
    def __init__(self, n_tree, aggregation_technique=None):
        self.n_tree = n_tree
        self.trees = []
        self.aggregation_technique = aggregation_technique
        if self.aggregation_technique == 'boosting':
            self.sum_correct_weights = np.full(self.n_tree, 1)
            self.sum_incorrect_weights = np.full(self.n_tree, 1)
        # use a set for classes to allow extension into multiclass classification
        self.classes = set()
        for i in range(self.n_tree):
            self.trees.append(MondrianTreeClassifier())

    def fit(self, X, y):
        for label in y:
            # compile dict of classes first
            self.classes |= {label}
        for i, x in enumerate(X):
            weight = 1
            for idx, tree in enumerate(self.trees):
                if self.aggregation_technique == 'boosting' and tree.root:
                    k = np.random.poisson(lam=weight)
                    for _ in range(k):
                        tree.fit(x, y[i])
                    prob = tree.predict_proba_single(x)
                    if prob.argmax() == y[i]:
                        self.sum_correct_weights[idx] += weight
                        weight *= (i / (2 * self.sum_correct_weights[idx]))
                    else:
                        self.sum_incorrect_weights[idx] += weight
                        weight *= (i / (2 * self.sum_incorrect_weights[idx]))
                elif self.aggregation_technique == 'bagging':
                    k = np.random.poisson(lam=1.0)
                    for _ in range(k):
                        tree.fit(x, y[i])
                else:
                    tree.fit(x, y[i])

    def predict_proba(self, X):
        if self.aggregation_technique == 'boosting':
            tree_res = np.zeros(len(X))
            scale = np.empty(self.n_tree)
            for idx, tree in enumerate(self.trees):
                eta = (self.sum_incorrect_weights[idx] / (self.sum_incorrect_weights[idx] + self.sum_correct_weights[idx]))
                beta = (eta / (1 - eta))
                scale[idx] = (np.log(1 / beta))
            return np.average([tree.predict_proba_batch(X) for tree in self.trees], axis=0, weights=scale)
        elif self.aggregation_technique == 'bagging':
            return np.average([tree.predict_proba_batch(X) for tree in self.trees], axis=0)
        else:
            return np.sum([tree.predict_proba_batch(X) for tree in self.trees], axis=0) / self.n_tree

    def predict_proba_single(self, x):
        #returns array of average probability outputs across all trees
        return np.sum([tree.predict_proba_single(x) if tree.root else [0.5, 0.5] for tree in self.trees], axis=0) / self.n_tree
