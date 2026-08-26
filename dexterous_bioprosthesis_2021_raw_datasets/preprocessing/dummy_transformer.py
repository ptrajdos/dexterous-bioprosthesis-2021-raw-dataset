"""Module providing a dummy (identity) transformer.

Returns deep copies of the input data without any modification,
useful as a no-op placeholder in scikit-learn pipelines.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy

class DummyTransformer(BaseEstimator, TransformerMixin):
    """Identity transformer that returns unmodified copies of the input."""

    def fit(self, X, y=None):
        """
        Does nothing
        """
        return self

    def transform(self, X, y=None):
        """Return a deep copy of the input data."""
        return deepcopy(X)
    
    def fit_transform(self, X, y=None, **fit_params):
        """Return a deep copy of the input data."""
        return deepcopy(X)
