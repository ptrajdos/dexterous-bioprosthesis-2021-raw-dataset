from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorEnergyEntropy(NPSignalExtractor):

    def fit(self, X, fs=1000):
        return super().fit(X)

    def _transform(self, X):

        sig_squares  =  X ** 2
        e_tot = np.sum(sig_squares, axis=0, keepdims=True)
        probs = np.divide(sig_squares, e_tot, where=e_tot>0)
        assert np.all(np.isnan(probs)) == False, "NaNs in probabilities calculation"
        logs = np.log2(probs + np.finfo(X.dtype).eps )
        assert np.all(np.isnan(logs)) == False, "NaNs in log calculation"
        entropy = -np.sum(probs * logs, axis=0).astype(X.dtype)
        assert np.all(np.isnan(entropy)) == False, "NaNs in entropy calculation"
        return entropy

    def attribs_per_column(self):
        return 1
