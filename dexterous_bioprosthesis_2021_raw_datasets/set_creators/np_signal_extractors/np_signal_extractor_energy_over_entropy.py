from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorEnergyOverEntropy(NPSignalExtractor):

    def fit(self, X):
        return super().fit(X)

    def _transform(self, X):

        sig_squares  =  X ** 2
        probs = sig_squares / np.sum(sig_squares, axis=0, keepdims=True)
        ch_energies = np.sum(sig_squares, axis=0)
        entropy = -np.sum(probs * np.log2(probs + np.finfo(float).eps ), axis=0).astype(X.dtype)
        eoe = ch_energies / (entropy + np.finfo(float).eps)
        eoe = eoe.astype(X.dtype)
        return eoe

    def attribs_per_column(self):
        return 1
