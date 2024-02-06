from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class UniformTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,extend_factor=0.1, element_fraction=1.0 ) -> None:
        super().__init__()
        self.extend_factor = extend_factor
        self.element_fraction = element_fraction

    
    def fit(self, X, y=None, **fit_params):
        """
        Does nothing
        """
        
        return self

    def transform(self, X, y=None):

        column_mins = np.min(X,axis=0)
        column_maxs = np.max(X,axis=0)

        n_rows, n_cols = X.shape

        n_effective_rows = int( np.max( (X.shape[0] * self.element_fraction , 1)))
        
        out = np.zeros((n_rows, n_cols))

        for col_idx in range(n_cols):
            ef_min = column_mins[col_idx] - self.extend_factor* np.abs(column_mins[col_idx])
            ef_max = column_maxs[col_idx] - self.extend_factor* np.abs(column_maxs[col_idx])
            out[:,col_idx] = np.random.uniform(low=ef_min, high =ef_max, size=(n_effective_rows))


        return out
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X,y,**fit_params).transform(X,y)