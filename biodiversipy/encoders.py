from sklearn.base import BaseEstimator, TransformerMixin

class SomeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, some_param):
        self.some_param = some_param

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''do some stuff on X'''
        return X
