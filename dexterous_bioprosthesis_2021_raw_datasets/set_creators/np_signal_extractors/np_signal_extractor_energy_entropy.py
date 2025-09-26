from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorEnergyEntropy(NPSignalExtractor):

    def fit(self, X):
        return super().fit(X)

    def _transform(self, X):

        energy  =  X ** 2
        probs = energy / np.sum(energy, axis=0, keepdims=True)
        entropy = -np.sum(probs * np.log2(probs + np.finfo(float).eps ), axis=0).astype(X.dtype)
        return entropy

    def attribs_per_column(self):
        return 1
