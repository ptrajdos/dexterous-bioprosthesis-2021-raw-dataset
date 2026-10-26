"""Module implementing energy entropy extraction.

Computes, per channel, the Shannon entropy (in bits) of the distribution of
signal energy over samples.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorEnergyEntropy(NPSignalExtractor):
    """Extractor that computes the energy entropy of each channel.

    Each sample's share of the channel's total energy is treated as a
    probability. Channels with zero total energy yield entropy 0.
    """

    def _transform(self, X):
        sig_squares = X ** 2
        e_tot = np.sum(sig_squares, axis=0, keepdims=True)
        probs = np.divide(
            sig_squares, e_tot, out=np.zeros_like(sig_squares), where=e_tot > 0
        )
        np.clip(probs, a_min=0, a_max=1, out=probs)
        logs = np.log2(probs, out=np.zeros_like(probs), where=probs > 0)
        entropy = -np.sum(probs * logs, axis=0)
        # + 0.0 turns -0.0 into 0.0
        return (entropy + 0.0).astype(X.dtype)

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
