import numpy as np

from .decision_tree_base import DecisionTreeBase

class DecisionTreeRegressor(DecisionTreeBase):
    def __init__(
        self, 
        criterion='mse', 
        max_depth=5, 
        leafs_num=-1, 
        max_objects_in_leaf_num=-1,
        min_impurity_decrease=0, 
        use_binning=True,
        verbose=False
    ):
        super(DecisionTreeRegressor, self).__init__(
            criterion=criterion,
            max_depth=max_depth,
            leafs_num= leafs_num,
            max_objects_in_leaf_num=max_objects_in_leaf_num,
            min_impurity_decrease=min_impurity_decrease,
            use_binning=use_binning,
            verbose=verbose
        )

    def _impurity_criterion(self, y):
        if self.criterion == 'mse':
            return np.var(y)
        elif self.criterion == 'mae':
            raise NotImplementedError()
        else:
            raise Exception('avalible criterions: MSE and MAE')

    def _decision_rule(self, y):
        if self.criterion == 'mse':
            return np.mean(y)
        elif self.criterion == 'mae':
            raise NotImplementedError()
        else:
            raise Exception('avalible criterions: MSE and MAE')