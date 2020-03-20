import numpy as np


def _check_shapes(x, y):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if x.shape == y.shape:
            return True
        else:
            return Exception('Arrays must have a same shape')
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) == len(y):
            return True
        else:
            raise Exception('Arrays must have a same shape')
    else:
        raise Exception('Support only Python List and NumPy array')