"""Module wrapping a set creator as a scikit-learn compatible transformer.

Adapts a :class:`SetCreator` to the scikit-learn transformer interface.
"""

from sklearn.base import BaseEstimator

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import SetCreator
from dexterous_bioprosthesis_2021_raw_datasets.tools.sklearn_compat import sklearn_transformer_tags
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_dummy import SetCreatorDummy


class SetCreatorTransformerWrapper(BaseEstimator):
    """Wrapper adapting a SetCreator to the scikit-learn transformer interface."""

    def __init__(self, set_creator: SetCreator = None) -> None:
        """Arguments:
        ---------
        set_creator -- SetCreator to wrap. None -- a new SetCreatorDummy

        """
        self.set_creator = set_creator

    def _get_set_creator(self) -> SetCreator:
        if self.set_creator is None:
            if not hasattr(self, "_default_set_creator"):
                self._default_set_creator = SetCreatorDummy()
            return self._default_set_creator
        return self.set_creator

    def fit(self, X, y=None):
        """Fit the transformer to the given data."""
        self._get_set_creator().fit(X, y)
        return self

    def transform(self, X):
        """Transform the given data."""
        Xt, yt, t = self._get_set_creator().transform(X)

        return Xt

    def fit_transform(self, X, y=None):
        """Fit and then transform the given data."""
        Xt, yt, t = self._get_set_creator().fit_transform(X, y)
        return Xt

    def get_feature_names_out(self, input_features=None):
        """Delegate to the wrapped SetCreator."""
        return self._get_set_creator().get_feature_names_out(input_features)

    def set_output(self, *, transform=None):
        """Set the output container of the wrapped SetCreator."""
        self._get_set_creator().set_output(transform=transform)
        return self

    def __sklearn_tags__(self):
        """scikit-learn >= 1.6 tags."""
        return sklearn_transformer_tags()

    def __sklearn_is_fitted__(self) -> bool:
        """Delegate to the wrapped SetCreator."""
        return self._get_set_creator().__sklearn_is_fitted__()
