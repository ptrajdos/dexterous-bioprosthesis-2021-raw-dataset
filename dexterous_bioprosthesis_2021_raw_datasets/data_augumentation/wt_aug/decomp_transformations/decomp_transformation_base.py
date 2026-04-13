from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.idecomp_transformation import (
    IDecompTransformation,
)


class DecompTransformationBase(IDecompTransformation):

    def __init__(self, random_state=10) -> None:
        self.random_state = random_state

    def _check_if_fitted(self):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def fit(self) -> IDecompTransformation:
        self._random_state = check_random_state(self.random_state)
        self._is_fitted = True
        return self
