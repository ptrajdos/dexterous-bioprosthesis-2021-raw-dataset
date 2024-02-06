
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class AttributeReorder(BaseEstimator, TransformerMixin):
    

    def fit(self, X, y=None):
        n_attribs = X.shape[1]
        self.indices = np.arange(n_attribs)
        np.random.shuffle(self.indices)

        return self

    def transform(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            return X.iloc[:,self.indices]
        
        return X[:,self.indices]
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)