import numpy as np

from .base import BaseModel

EPSILON = 1e-10
CATEGORICAL_CONST = 100
BINS_NUMBER = 100

NUMERICAL_KEY = 'numerical'
CATEGORICAL_KEY = 'categorical'

class DecisionTreeBase(BaseModel):
    def __init__(
        self, 
        criterion='mse', 
        max_depth=5, 
        leafs_num=-1, 
        max_objects_in_leaf_num=-1,
        min_impurity_decrease=0, 
        use_binning=True,
        use_rsm=False,
        verbose=False
    ):
        super(DecisionTreeBase, self).__init__(verbose)
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_objects_in_leaf_num = max_objects_in_leaf_num
        self.min_impurity_decrease = min_impurity_decrease        
        self.use_binning = use_binning
        self.use_rsm = use_rsm
        
        if leafs_num != -1:
            self.leafs_num = leafs_num
        else:
            self.leafs_num = (max_depth + 1) ** 2        

        self._features_types = []
        self._features_num = 0
        self._samples_num = 0
        self._tree = {}

    def _impurity_criterion(self, y):
        raise NotImplementedError()

    def _decision_rule(self, y):
        raise NotImplementedError()   

    def _infer_features_types(self, X):
        self._samples_num = X.shape[0]
        self._features_num = X.shape[1]

        for i in range(self._features_num):
            if len(np.unique(X[:,i])) < self._samples_num / CATEGORICAL_CONST:
                # TODO: realize categorical features
                # self._features_types.append(CATEGORICAL_KEY)
                self._features_types.append(NUMERICAL_KEY)
            else:
                self._features_types.append(NUMERICAL_KEY)

    def _get_left_child_index(self, parent_index):
        return parent_index * 2 + 1

    def _get_right_child_index(self, parent_index):
        return parent_index * 2 + 2            

    def _get_split(self, X, y):
        split_feature_index, split_feature_value = -1, -1
        final_left_indexes, final_right_indexes = [], []
        is_leaf, decision = False, None

        best_criterion_value = -1

        if X.shape[0] < self.max_objects_in_leaf_num:
            is_leaf = True
            decision = self._decision_rule(y)
            return -1, -1, -1, -1, is_leaf, decision

        for j in range(self._features_num):
            # little hack for Random Space Method
            if self.use_rsm and np.random.random() > 1 / np.sqrt(self._features_num):
                continue            
            if self._features_types[j] == NUMERICAL_KEY:
                # hack to solve problems with binning when we
                # have a small number of objects
                if  self.use_binning and len(np.unique(X[:,j])) > BINS_NUMBER * 10:
                    bins = np.linspace(min(X[:,j]), max(X[:,j]), BINS_NUMBER)
                    digitized = np.digitize(X[:,j], bins)
                    bin_means = [X[:,j][digitized == i].mean() for i in range(1, len(bins))]
                else:
                    bin_means = np.unique(X[:,j])

                for bin_value in bin_means:
                    left_indexes = np.where(X[:,j] < bin_value)[0]
                    right_indexes = np.where(X[:,j] >= bin_value)[0]

                    if (len(left_indexes) == 0) or (len(right_indexes) == 0):
                        continue

                    parent_criterion = self._impurity_criterion(y)
                    left_criterion = self._impurity_criterion(y[left_indexes])
                    right_criterion = self._impurity_criterion(y[right_indexes])

                    if parent_criterion < EPSILON:
                        # if variance of parent is equal to zero
                        is_leaf = True
                        decision = self._decision_rule(y)
                        return -1, -1, -1, -1, is_leaf, decision

                    criterion_value = (
                        parent_criterion 
                        - (len(left_indexes) / X.shape[0]) * left_criterion 
                        - (len(right_indexes) / X.shape[0]) * right_criterion
                    )

                    if criterion_value > best_criterion_value:
                        best_criterion_value = criterion_value
                        split_feature_index = j
                        split_feature_value = bin_value

                        final_left_indexes = left_indexes
                        final_right_indexes = right_indexes

            else:
                raise NotImplementedError()

        if best_criterion_value == -1:
            is_leaf = True
            decision = self._decision_rule(y)
            return -1, -1, -1, -1, is_leaf, decision

        if best_criterion_value <= (self.min_impurity_decrease + EPSILON):
            is_leaf = True
            decision = self._decision_rule(y)
            return -1, -1, -1, -1, is_leaf, decision            
        
        return split_feature_index, split_feature_value, final_left_indexes, final_right_indexes, \
            is_leaf, decision

    def _get_available_nodes(self):
        available_nodes = []

        for node_index, node_data in self._tree.items():
            if (
                node_data['is_leaf'] == False and 
                self._get_left_child_index(node_index) not in self._tree and 
                self._get_right_child_index(node_index) not in self._tree
            ):
                available_nodes.append(node_index)

        return available_nodes

    def _make_leaf(self, node_index, y):
        samples_indexes = self._tree[node_index]['left_indexes']
        samples_indexes = np.append(samples_indexes, self._tree[node_index]['right_indexes'])

        self._tree[node_index]['is_leaf'] = True
        self._tree[node_index]['decision'] = self._decision_rule(y[samples_indexes])

    def _count_leafs(self):
        count = 0
        for node_index, node_data in self._tree.items():
            if (
                self._get_left_child_index(node_index) not in self._tree and
                self._get_right_child_index(node_index) not in self._tree
            ):
                count += 1
        return count

    def fit(self, X, y):
        self._check_dimensions(X, y)
        self._infer_features_types(X)

        split_feature_index, split_feature_value, left_indexes, right_indexes, \
            is_leaf, decision = self._get_split(X, y)

        self._tree[0] = {
            'split_feature_index': split_feature_index,
            'split_feature_value': split_feature_value, 
            'left_indexes': left_indexes, 
            'right_indexes': right_indexes, 
            'is_leaf': is_leaf,
            'decision': decision
        }

        available_nodes = self._get_available_nodes()

        while len(available_nodes) != 0:
            for node_index in available_nodes:
                # using epsilon to overcome numerical problems
                if np.log2(node_index + EPSILON) + 1 >= self.max_depth:
                    self._make_leaf(node_index, y)
                    continue

                if self._count_leafs() >= self.leafs_num:
                    for candidate_node in available_nodes:
                        self._make_leaf(candidate_node, y)
                    break

                parent_left_indexes = self._tree[node_index]['left_indexes']
                parent_right_indexes = self._tree[node_index]['right_indexes']

                mapping = {i: parent_left_indexes[i] for i in range(len(parent_left_indexes))}

                split_feature_index, split_feature_value, left_indexes, right_indexes, \
                    is_leaf, decision = self._get_split(X[parent_left_indexes], y[parent_left_indexes])

                if not type(left_indexes) is int and not type(right_indexes) is int:
                    left_indexes_mapped= [mapping[i] for i in left_indexes]
                    right_indexes_mapped = [mapping[i] for i in right_indexes]
                else:
                    left_indexes_mapped = left_indexes
                    right_indexes_mapped = right_indexes

                self._tree[self._get_left_child_index(node_index)] = {
                    'split_feature_index': split_feature_index,
                    'split_feature_value': split_feature_value, 
                    'left_indexes': left_indexes_mapped, 
                    'right_indexes': right_indexes_mapped, 
                    'is_leaf': is_leaf,
                    'decision': decision
                }

                mapping = {i: parent_right_indexes[i] for i in range(len(parent_right_indexes))}

                split_feature_index, split_feature_value, left_indexes, right_indexes, \
                    is_leaf, decision = self._get_split(X[parent_right_indexes], y[parent_right_indexes])

                if not type(left_indexes) is int and not type(right_indexes) is int:
                    left_indexes_mapped = [mapping[i] for i in left_indexes]
                    right_indexes_mapped = [mapping[i] for i in right_indexes]
                else:
                    left_indexes_mapped = left_indexes
                    right_indexes_mapped = right_indexes

                self._tree[self._get_right_child_index(node_index)] = {
                    'split_feature_index': split_feature_index,
                    'split_feature_value': split_feature_value, 
                    'left_indexes': left_indexes_mapped, 
                    'right_indexes': right_indexes_mapped, 
                    'is_leaf': is_leaf,
                    'decision': decision
                }

            available_nodes = self._get_available_nodes()

    def predict(self, X):
        if len(X.shape) == 1:
            X = [X]

        predictions = []
        for x in X:
            current_node_index = 0
            current_node = self._tree[current_node_index]

            while current_node['is_leaf'] != True:
                split_feature_index = current_node['split_feature_index']
                split_feature_value = current_node['split_feature_value']

                if x[split_feature_index] >= split_feature_value:
                    current_node_index = self._get_right_child_index(current_node_index)
                else:
                    current_node_index = self._get_left_child_index(current_node_index)
                
                current_node = self._tree[current_node_index]

            predictions.append(current_node['decision'])
        
        return np.array(predictions)