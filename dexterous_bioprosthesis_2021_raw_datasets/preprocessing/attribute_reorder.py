"""Module providing a random attribute (column) reordering transformer.

Shuffles the column order of a dataset randomly, useful for testing
model sensitivity to feature ordering.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class AttributeReorder(BaseEstimator, TransformerMixin):
    """Transformer that randomly reorders columns (attributes) of a dataset.

    On :meth:`fit`, a random permutation of column indices is generated.
    On :meth:`transform`, columns are reordered according to that permutation.
    Supports both NumPy arrays and pandas DataFrames.
    """
    

    def fit(self, X, y=None):
        """Generate a random column permutation.

        Args:
            X: Input data with shape ``(n_samples, n_features)``.
            y: Ignored.

        Returns:
            The fitted transformer.
        """
        n_attribs = X.shape[1]
        self.indices = np.arange(n_attribs)
        np.random.shuffle(self.indices)

        return self

    def transform(self, X, y=None):
        """Reorder columns according to the fitted permutation.

        Args:
            X: Input data (NumPy array or pandas DataFrame).
            y: Ignored.

        Returns:
            Data with reordered columns.
        """

        if isinstance(X, pd.DataFrame):
            return X.iloc[:,self.indices]
        
        return X[:,self.indices]
    
    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform in a single step."""
        return self.fit(X).transform(X)