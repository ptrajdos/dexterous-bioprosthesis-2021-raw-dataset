from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorZcr(NPSignalExtractor):

    def fit(self, X):
        return super().fit(X)

    def _transform(self, X):
        data_sign = np.sign(X - np.mean(X, axis=0))
        grads = np.diff(data_sign, axis=0)
        zcr = np.mean(np.abs(grads), axis=0)
        return zcr

    def attribs_per_column(self):
        return 1
