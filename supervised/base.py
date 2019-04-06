class BaseModel():
    def __init__(self, verbose=False):
        self.X = None
        self.y = None
        self.verbose = verbose

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _check_dimensions(self, X, y):
        if len(X.shape) != 2:
            raise Exception(X.shape)
        if len(y.shape) != 1:
            raise Exception(y.shape)