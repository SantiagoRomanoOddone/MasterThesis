
from catboost import CatBoostRegressor as CatBoostModel


class CatBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None
    
    def fit(self, X, y):
        self.model = CatBoostModel(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

