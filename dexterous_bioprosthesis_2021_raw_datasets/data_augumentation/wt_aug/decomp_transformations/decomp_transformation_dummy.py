from copy import deepcopy
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_base import (
    DecompTransformationBase,
)


class DecompTransformationDummy(DecompTransformationBase):

    def transform(self, decompositions: list):
        self._check_if_fitted()
        return deepcopy(decompositions)
