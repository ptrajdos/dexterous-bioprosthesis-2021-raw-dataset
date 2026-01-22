from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorEnergyOverEntropy(NPSignalExtractor):

    def fit(self, X, fs=1000):
        return super().fit(X)

    def _transform(self, X):
        dtype = X.dtype

        sig_squares = X ** 2
        ch_energies = np.sum(sig_squares, axis=0)

        eoe = np.zeros(X.shape[1], dtype=dtype)

        nonzero = ch_energies > 0

        if np.any(nonzero):
            probs = np.zeros_like(sig_squares, dtype=float)
            probs[:, nonzero] = (
                sig_squares[:, nonzero] /
                ch_energies[nonzero]
            )

            entropy = np.zeros(X.shape[1], dtype=float)
            entropy[nonzero] = -np.sum(
                probs[:, nonzero] * np.log2(probs[:, nonzero]),
                axis=0,
                where=probs[:, nonzero] > 0
            )

            valid = nonzero & (entropy > 0)
            eoe[valid] = ch_energies[valid] / entropy[valid]

        return eoe.astype(dtype)

    def attribs_per_column(self):
        return 1
