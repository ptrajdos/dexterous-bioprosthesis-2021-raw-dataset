"""Module implementing Zero Crossing Rate (ZCR) extraction.

Computes, per channel, the rate of crossings of the channel mean.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorZcr(NPSignalExtractor):
    """Extractor that computes the Zero Crossing Rate."""

    def _transform(self, X):
        data_sign = np.sign(X - np.mean(X, axis=0))
        grads = np.diff(data_sign, axis=0)
        # A sign change -1 -> 1 has |grad| == 2; halve so one crossing counts once.
        zcr = np.mean(np.abs(grads), axis=0) / 2
        return zcr

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
