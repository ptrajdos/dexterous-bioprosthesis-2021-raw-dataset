from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np
from numba import jit


# @jit(nopython=True)
def ssc_int(input):
    re = np.copy(input).astype(input.dtype)
    re[re <= 0] = 0.0
    re[re > 0] = 1.0
    return re


class NpSignalExtractorSsc(NPSignalExtractor):

    def fit(self, X):
        return super().fit(X)

    def _transform(self, X):
        SSC = np.mean(
            ssc_int(-np.diff(X, axis=0, prepend=1)[1:-1] * np.diff(X, axis=0)[1:]),
            axis=0,
        )
        return SSC.astype(X.dtype)

    def attribs_per_column(self):
        return 1
