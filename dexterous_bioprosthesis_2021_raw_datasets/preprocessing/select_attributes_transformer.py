"""Module providing a column (attribute) selection transformer.

Selects a subset of columns from the input data by index, supporting
both NumPy arrays and pandas DataFrames.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class SelectAttributesTransformer(BaseEstimator, TransformerMixin):
    """Transformer that selects specific columns by index.

    Args:
        column_indices: List of column indices to retain.

    """

    def __init__(self, column_indices=[0]) -> None:
        """Selects a single attribute (column) from datases

        Arguments:
        column_number:int -- column to select

        """
        super().__init__()
        self.column_indices = column_indices


    def fit(self, X, y=None):
        """Does nothing
        """
        return self

    def transform(self, X, y=None):
        """Select the configured columns from the input data.

        Args:
            X: Input data (NumPy array or pandas DataFrame).
            y: Ignored.

        Returns:
            Data with only the selected columns.

        """
        if isinstance(X, pd.DataFrame):
            return X.iloc[:,self.column_indices]

        return X[:,self.column_indices]