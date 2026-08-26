"""Module wrapping a set creator as a scikit-learn compatible transformer.

Adapts a :class:`SetCreator` to the scikit-learn transformer interface.
"""

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import SetCreator
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_dummy import SetCreatorDummy


class SetCreatorTransformerWrapper:
    """Wrapper adapting a SetCreator to the scikit-learn transformer interface."""
    def __init__(self, set_creator:SetCreator=SetCreatorDummy()) -> None:
        self.set_creator = set_creator

    def fit(self, X, y=None):
        """Fit the transformer to the given data."""
        self.set_creator.fit(X,y)
        return self
    
    def transform(self, X):
        """Transform the given data."""
        Xt, yt, t = self.set_creator.transform(X)

        return Xt
    
    def fit_transform(self, X, y=None):

        """Fit and then transform the given data."""
        Xt, yt, t = self.set_creator.fit_transform(X, y)
        return Xt