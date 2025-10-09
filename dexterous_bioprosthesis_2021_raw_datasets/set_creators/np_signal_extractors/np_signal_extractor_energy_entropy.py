from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorEnergyEntropy(NPSignalExtractor):

    def fit(self, X, fs=1000):
        return super().fit(X)

    def _transform(self, X):

        sig_squares  =  X ** 2
        probs = sig_squares / np.sum(sig_squares, axis=0, keepdims=True)
        entropy = -np.sum(probs * np.log2(probs + np.finfo(float).eps ), axis=0).astype(X.dtype)
        return entropy

    def attribs_per_column(self):
        return 1
