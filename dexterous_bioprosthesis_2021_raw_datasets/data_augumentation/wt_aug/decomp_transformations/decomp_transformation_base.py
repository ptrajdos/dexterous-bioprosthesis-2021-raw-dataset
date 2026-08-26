"""Module providing the base implementation for decomposition transformations.

Contains :class:`DecompTransformationBase`, which adds random state
management and fitted-state checking to the :class:`IDecompTransformation`
interface.
"""
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.idecomp_transformation import (
    IDecompTransformation,
)


class DecompTransformationBase(IDecompTransformation):
    """Base class for decomposition coefficient transformations.

    Provides random state initialisation and fitted-state validation.
    Subclasses must implement :meth:`transform`.

    Args:
        random_state: Random seed for reproducibility.
    """

    def __init__(self, random_state=10) -> None:
        self.random_state = random_state

    def _check_if_fitted(self):
        """Raise :class:`NotFittedError` if the transformation has not been fitted."""
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def fit(self) -> IDecompTransformation:
        """Initialise the random state and mark as fitted.

        Returns:
            The fitted transformation instance.
        """
        self._random_state = check_random_state(self.random_state)
        self._is_fitted = True
        return self
