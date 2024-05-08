import numpy as np
from .classifier import Classifier

class Node(object):
    def __init__(self, min_list, max_list, tau, is_leaf, stat, parent=None, delta=None, xi=None):
        """
        Initialize a node in a Mondrian tree.

        Parameters:
        - min_list: List of minimum values for each dimension in the input space.
        - max_list: List of maximum values for each dimension in the input space.
        - tau: Split time (the time when this node was split).
        - is_leaf: Flag indicating whether the node is a leaf node.
        - stat: Statistics associated with the node (e.g., for leaf nodes, it represents data statistics).
        - parent: Reference to the parent node.
        - delta: Parameter representing the maximum age of the node (time until the node can be split again).
        - xi: Parameter representing the budget available for the node (maximum range of the split).
        """
        self.parent = parent
        self.tau = tau
        self.is_leaf = is_leaf
        self.min_list = min_list
        self.max_list = max_list
        self.delta = delta
        self.xi = xi
        self.left = None
        self.right = None
        self.stat = stat

    def update_leaf(self, x, label):
        """
        Update the statistics of a leaf node.

        Parameters:
        - x: Input data point.
        - label: Label associated with the input data point.
        """
        self.stat.add(x, label)

    def update_internal(self):
        """
        Update the statistics of an internal node by merging the statistics of its children.
        """
        self.stat = self.left.stat.merge(self.right.stat)

    def get_parent_tau(self):
        """
        Get the split time of the parent node.

        Returns:
        - Split time of the parent node or 0.0 if there is no parent (root node).
        """
        if self.parent is None:
            return 0.0
        return self.parent.tau

    def __repr__(self):
        """
        String representation of the Node object.

        Returns:
        - A string representation containing essential information about the node.
        """
        return "<mondrianforest.Node tau={} min_list={} max_list={} is_leaf={}>".format(
            self.tau,
            self.min_list,
            self.max_list,
            self.is_leaf,
        )

class MondrianTree(object):
    def __init__(self):
        """
       Initialize a Mondrian Tree.

       Attributes:
       - root: Root node of the tree.
       - classes: Set of unique classes observed in the data.
       """
        self.root = None
        self.classes = set()
        self.classes.add(0)
        self.classes.add(1)

    def create_leaf(self, x, label, parent):
        """
        Create a new leaf node.

        Parameters:
        - x: Input data point.
        - label: Label associated with the input data point.
        - parent: Parent node of the new leaf node.

        Returns:
        - The created leaf node.
        """
        leaf = Node(
            min_list=x.copy(),
            max_list=x.copy(),
            is_leaf=True,
            stat=Classifier(),
            tau=1e9,
            parent=parent,
        )
        leaf.update_leaf(x, label)
        return leaf

    def extend_mondrian_block(self, node, x, label):
        """
        Extend the Mondrian block by recursively splitting or updating the node.

        Parameters:
        - node: Current node being considered.
        - x: Input data point.
        - label: Label associated with the input data point.

        Returns:
        - The root of the resulting sub-tree.
        """
        e_min = np.maximum(node.min_list - x, 0)
        e_max = np.maximum(x - node.max_list, 0)
        e_sum = e_min + e_max
        rate = np.sum(e_sum) + 1e-9
        E = np.random.exponential(1.0/rate)
        if node.get_parent_tau() + E < node.tau:
            # splitting threshold is exceeded, so we split along some delta that maximizes a sampled expansion
            e_sample = np.random.rand() * np.sum(e_sum)
            delta = (e_sum.cumsum() > e_sample).argmax()
            if x[delta] > node.min_list[delta]:
                # if x is above the lower bound of this node, we pick a criterion from the range min - x
                xi = np.random.uniform(node.min_list[delta], x[delta])
            else:
                # otherwise, our criterion comes from x - max
                xi = np.random.uniform(x[delta], node.max_list[delta])
            # create a new parent node, essentially a clone of node
            parent = Node(
                min_list=np.minimum(node.min_list, x),
                max_list=np.maximum(node.max_list, x),
                is_leaf=False,
                stat=Classifier(),
                tau=node.get_parent_tau() + E,
                parent=node.parent,
                delta=delta,
                xi=xi,
            )
            # create a new sibling node as a leaf for the result of classification
            sibling = self.create_leaf(x, label, parent=parent)
            # assign children including old node to new parent
            if x[parent.delta] <= parent.xi:
                parent.left = sibling
                parent.right = node
            else:
                parent.left = node
                parent.right = sibling
            node.parent = parent
            # merge stats of node and sibling under parent, return new root (parent)
            parent.update_internal()
            return parent
        else:
            # split threshold is not exceeded, so we add x to this node's stats, updating max and min
            node.min_list = np.minimum(x, node.min_list)
            node.max_list = np.maximum(x, node.max_list)
            if not node.is_leaf:
                # recursively update children based on classification of x using xi

                if x[node.delta] <= node.xi:
                    node.left = self.extend_mondrian_block(node.left, x, label)
                else:
                    node.right = self.extend_mondrian_block(node.right, x, label)
                    # since we didn't split, we update this node by merging those below it
                node.update_internal()
            else:
                # add x to this leaf's stats
                node.update_leaf(x, label)
            # return node as all extensions have been processed internally
            return node

    def partial_fit(self, x, y):
        """
        Incrementally update the Mondrian Tree with new data.

        Parameters:
        - X: Input data.
        - y: Label associated with the input data.
        """
        self.classes |= {y}

        if self.root is None:
            self.root = self.create_leaf(x, y, parent=None)
        else:
            self.root = self.extend_mondrian_block(self.root, x, y)

    def fit(self, X, y):
        """
        Fit the Mondrian Tree with a new data point.

        Parameters:
        - X: Input data point.
        - y: Label associated with the input data.
        """
        self.partial_fit(X, y)

    def _predict(self, x, node, p_not_separeted_yet):

        """
        Recursively predict probabilities for a given input data point.

        Parameters:
        - x: Input data point.
        - node: Current node being considered.
        - p_not_separated_yet: Probability of not being separated until the current node.

        Returns:
        - ClassifierResult object containing prediction probabilities.
        """
        d = node.tau - node.get_parent_tau()
        eta = np.sum(np.maximum(x-node.max_list, 0) + np.maximum(node.min_list - x, 0))
        p = 1.0 - np.exp(-d*eta)
        result = node.stat.create_result(x, p_not_separeted_yet * p)

        if node.is_leaf:
            w = p_not_separeted_yet * (1.0 - p)
            return result.merge(node.stat.create_result(x, w))
        if x[node.delta] <= node.xi:
            child_result = self._predict(x, node.left, p_not_separeted_yet*(1.0-p))
        else:
            child_result = self._predict(x, node.right, p_not_separeted_yet*(1.0-p))
        return result.merge(child_result)


class MondrianTreeClassifier(MondrianTree):
    def __init__(self):
        MondrianTree.__init__(self)

    def predict_proba_batch(self, X):
        """
        Predict class probabilities with a data point.
        Parameters:
        - x: Input data point.
        Returns:
        - List object indexed by class probability
        """
        res = []
        for x in X:
            prob = self._predict(x, self.root, 1.0).get()
            res.append(np.array([prob[l] for l in self.classes]))
        return res

    def predict_proba_single(self, x):
        prob = self._predict(x, self.root, 1.0).get()
        return np.array([prob[l] for l in self.classes])
