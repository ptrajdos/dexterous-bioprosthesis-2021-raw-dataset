"""Module providing a uniform random data generator transformer.

Replaces input data with uniformly distributed random values within
the column-wise min/max range (optionally extended), useful for
generating synthetic baseline data.
"""
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class UniformTransformer(BaseEstimator, TransformerMixin):
    """Transformer that replaces data with uniform random values.

    Generates random values per column within the observed min/max range,
    optionally extended by ``extend_factor``.

    Args:
        extend_factor: Fraction by which to extend the column value range.
        element_fraction: Fraction of rows to generate (minimum 1).

    """

    def __init__(self,extend_factor=0.1, element_fraction=1.0 ) -> None:
        super().__init__()
        self.extend_factor = extend_factor
        self.element_fraction = element_fraction

    
    def fit(self, X, y=None, **fit_params):
        """Does nothing
        """
        return self

    def transform(self, X, y=None):
        """Generate uniform random data matching the input's column ranges.

        Args:
            X: Input data with shape ``(n_samples, n_features)``.
            y: Ignored.

        Returns:
            Array of uniform random values with the same shape as *X*.

        """
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
        """Fit and transform in a single step."""
        return self.fit(X,y,**fit_params).transform(X,y)