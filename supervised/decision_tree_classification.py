import numpy as np
from scipy.stats import mode
from .decision_tree_base import DecisionTreeBase

class DecisionTreeClassifier(DecisionTreeBase):
    def __init__(
        self, 
        criterion='gini', 
        max_depth=5, 
        leafs_num=-1, 
        max_objects_in_leaf_num=-1,
        min_impurity_decrease=0, 
        use_binning=True,
        verbose=False
    ):
        super(DecisionTreeClassifier, self).__init__(
            criterion,
            max_depth,
            leafs_num,
            max_objects_in_leaf_num,
            min_impurity_decrease,
            use_binning,
            verbose
        )

    def _impurity_criterion(self, y):
        if self.criterion == 'gini':
            gini = 0
            
            for y_ in np.unique(y):
                p_k = len(y[y == y_]) / len(y)
                gini += p_k * (1 - p_k)

            return gini

        elif self.criterion == 'entropy':
            entropy = 0
            
            for y_ in np.unique(y):
                p_k = len(y[y == y_]) / len(y)
                entropy += p_k * np.log2(p_k)

            return -entropy

        else:
            raise Exception('avalible criterions: GINI and ENTROPY')

    def _decision_rule(self, y):
        if self.criterion == 'gini' or self.criterion == 'entropy':
            return mode(y)[0][0]
        else:
            raise Exception('avalible criterions: GINI and ENTROPY')