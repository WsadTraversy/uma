import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, is_leaf_node=False):
        self._feature = feature
        self._threshold = threshold
        self._left = left
        self._left_before_removal = None
        self._right = right
        self._right_before_removal = None
        self._value = value
        self._is_leaf_node = is_leaf_node

    def feature(self):
        return self._feature

    def threshold(self):
        return self._threshold

    def left(self):
        return self._left
    
    def right(self):
        return self._right

    def value(self):
        return self._value

    def is_leaf_node(self):
        return self._is_leaf_node
    
    def delete_children_nodes(self, value=None):
        if not self.is_leaf_node():
            self._left_before_removal = self._left
            self._left_value = self._left._value
            self._right_before_removal = self._right
            self._right_value = self._left._value
            self._left = None
            self._right = None
            self._value = value
            self._is_leaf_node = True

    def restore(self):
        self._left = self._left_before_removal
        self._right = self._right_before_removal
        self._value = None
        self._left_before_removal = None  
        self._right_before_removal = None   
        self._is_leaf_node = False


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self._min_samples_split=min_samples_split
        self._max_depth=max_depth
        self._number_of_features=None
        self._root=None
    
    def fit(self, X, y):
        self._number_of_features = X.shape[1]
        self._root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self._max_depth or n_labels == 1 or n_samples < self._min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf_node=True)
        
        features_idxs = np.random.choice(n_feats, self._number_of_features, replace=False)

        # find best split
        best_feature, best_thresh = self._best_split(X, y, features_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs, :], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs, :], depth+1)
        leaf_value = self._most_common_label(y)
        return Node(best_feature, best_thresh, left, right, value=leaf_value)

    def _best_split(self, X, y, features_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for idx in features_idxs:
            X_column = X[:, idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(X_column, y, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, X_column, y, threshold):
        # parent gini index
        parent_gini = self._calculate_gini_index(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. gini index of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._calculate_gini_index(y[left_idxs]), self._calculate_gini_index(y[right_idxs])
        child_gini = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_gini - child_gini
        return information_gain
        
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _calculate_gini_index(self, y):
        hist = np.bincount(np.hstack(y))
        ps = hist / len(y)
        return 1-np.sum([p**2 for p in ps if p>0])
       
    def _most_common_label(self, y):
        # counter = Counter(y)
        # value = counter.most_common(1)[0][0]
        counter = np.bincount(np.hstack(y))
        value = np.argmax(counter)
        return value
 
    def cut_tree(self, X_valid, y_valid):
        self._cut_tree_left(X_valid, y_valid, was_left=False, depth=0, start=True, recursion_depth=900)
        self._cut_tree_right(X_valid, y_valid, was_right=False, depth=0, start=True, recursion_depth=900)

    def _cut_tree_left(self, X_valid, y_valid, was_left, depth, start, recursion_depth):
        recursion_depth -= 1
        parent_node = None
        current_node_left = self._root
        current_node_right = None
        depth -= 2
        depth_counter = 0
        while True:
            if current_node_left.is_leaf_node():
                break
            depth_counter += 1
            depth -= 1
            if depth == 0 and not start and was_left:
                current_node_right = current_node_left.right() if not current_node_right else current_node_right.right()
                parent_node = current_node_right
                current_node_right = current_node_right.right()
            else: 
                parent_node = current_node_left
                current_node_left = current_node_left.left()

        if parent_node:
            current_node_right=parent_node.right() if not current_node_right else current_node_right

            acc1 = self._accuracy(X_valid, y_valid)
            if parent_node.value() == current_node_left.value():
                parent_node.delete_children_nodes(current_node_left.value())
            elif parent_node.value() == current_node_right.value():
                parent_node.delete_children_nodes(current_node_right.value())
            elif current_node_left.value() == current_node_right.value():
                parent_node.delete_children_nodes(current_node_left.value())
            elif current_node_left.value() is None:
                parent_node.delete_children_nodes(current_node_right.value())
            elif current_node_right.value() is None:
                parent_node.delete_children_nodes(current_node_left.value())
            acc2 = self._accuracy(X_valid, y_valid)
            was_left = True if not was_left else False
            if acc2 < acc1:
                parent_node.restore()
            else:
                if recursion_depth != 0:
                    self._cut_tree_left(X_valid, y_valid, was_left, depth=depth_counter, start=False, recursion_depth=recursion_depth)

    def _cut_tree_right(self, X_valid, y_valid, was_right, depth, start, recursion_depth):
        recursion_depth -= 1
        parent_node = None
        current_node_left = None
        current_node_right = self._root
        depth -= 2
        depth_counter = 0
        while True:
            if current_node_right.is_leaf_node():
                break
            depth_counter += 1
            depth -= 1
            if depth == 0 and not start and was_right:
                current_node_left = current_node_right.left() if not current_node_left else current_node_left.left()
                parent_node = current_node_left
                current_node_left = current_node_left.left()
            else: 
                parent_node = current_node_right
                current_node_right = current_node_right.right()

        if parent_node:
            current_node_left=parent_node.left() if not current_node_left else current_node_left

            acc1 = self._accuracy(X_valid, y_valid)
            if parent_node.value() == current_node_left.value():
                parent_node.delete_children_nodes(current_node_left.value())
            elif parent_node.value() == current_node_right.value():
                parent_node.delete_children_nodes(current_node_right.value())
            elif current_node_left.value() == current_node_right.value():
                parent_node.delete_children_nodes(current_node_left.value())
            elif current_node_left.value() is None:
                parent_node.delete_children_nodes(current_node_right.value())
            elif current_node_right.value() is None:
                parent_node.delete_children_nodes(current_node_left.value())
            acc2 = self._accuracy(X_valid, y_valid)
            was_right = True if not was_right else False
            if acc2 < acc1:
                parent_node.restore()
            else:
                if recursion_depth != 0:
                    self._cut_tree_right(X_valid, y_valid, was_right, depth=depth_counter, start=False, recursion_depth=recursion_depth)

    def _accuracy(self, X_valid, y_valid):
        value = 0
        predictions = self.predict(X_valid)
        for idx, el in enumerate(y_valid):
            value += 1 if el == predictions[idx] else 0 
        accuracy = value / len(y_valid)
        return accuracy
        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self._root) for x in X])

    def _traverse_tree(self, x, node:Node):
        if node.is_leaf_node():
            return node.value()

        if x[node.feature()] <= node.threshold():
            return self._traverse_tree(x, node.left())
        return self._traverse_tree(x, node.right())


class RandomForest:
    def __init__(self, n_trees=10, max_depth=100, min_samples_split=2):
        self._n_trees = n_trees
        self._max_depth=max_depth
        self._min_samples_split=min_samples_split
        self._trees = []
    
    def show_number_of_trees(self):
        return len(self._trees)

    def fit(self, X, y):
        self._trees = []
        for _ in range(self._n_trees):
            tree = DecisionTree(max_depth=self._max_depth,
                            min_samples_split=self._min_samples_split)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self._trees.append(tree)

    def cut_forest(self, X_valid, y_valid):
        best_acc = self._accuracy(X_valid, y_valid)
        trees = self._trees
        for tree in trees:
            self._trees.remove(tree)
            acc = self._accuracy(X_valid, y_valid)
            if acc >= best_acc:
                best_acc = acc
            else :
                self._trees.append(tree)

    def cut_tree(self, X_valid, y_valid):
        for tree in self._trees:
            tree.cut_tree(X_valid, y_valid)

    def _accuracy(self, X_valid, y_valid):
        value = 0
        predictions = self.predict(X_valid)
        for idx, el in enumerate(y_valid):
            value += 1 if el == predictions[idx] else 0 
        accuracy = value / len(y_valid)
        return accuracy

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = np.bincount(np.hstack(y))
        value = np.argmax(counter)
        return value

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self._trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions


class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self._learning_rate = learning_rate
        self._n_iterations = n_iterations
        self._weights = 0
        self._bias = 0
    
    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features)
        self._bias = 0

    
        for _ in range(self._n_iterations):
            linear_pred = np.dot(X, self._weights) + self._bias
            predictions = self._sigmoid(linear_pred)
            
            dw = (1/n_samples) * np.dot(X.T, (predictions-y.ravel()))
            db = (1/n_samples) * np.sum(predictions-y.ravel())
    
            self._weights = self._weights - self._learning_rate*dw
            self._bias = self._bias * self._learning_rate*db

    def predict(self, X):
        linear_pred = np.dot(X, self._weights) + self._bias
        prediction = self._sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in prediction]

        return class_pred