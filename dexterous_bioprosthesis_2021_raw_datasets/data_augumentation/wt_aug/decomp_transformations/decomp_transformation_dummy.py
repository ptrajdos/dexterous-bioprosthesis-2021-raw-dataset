"""Module providing a dummy (identity) decomposition transformation.

Returns deep copies of the input coefficients without modification,
useful as a baseline or no-op placeholder.
"""
from copy import deepcopy
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_base import (
    DecompTransformationBase,
)


class DecompTransformationDummy(DecompTransformationBase):
    """Identity transformation that returns unmodified coefficient copies."""

    def transform(self, decompositions: list):
        """Return a deep copy of the decomposition coefficients.

        Args:
            decompositions: List of wavelet coefficient arrays.

        Returns:
            Deep copy of the input coefficients.
        """
        self._check_if_fitted()
        return deepcopy(decompositions)
