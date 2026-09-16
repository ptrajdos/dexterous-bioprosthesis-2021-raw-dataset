"""Module implementing Zero Crossing Rate (ZCR) extraction.

Counts the number of zero crossings in each channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorNWL(NPSignalExtractor):
    """Extractor that computes the Zero Crossing Rate."""

    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        return super().fit(X)

    def _transform(self, X):
        nwl = np.sum(np.abs(np.diff(X, axis=0)), axis=0) / (len(X) - 1)
        return nwl

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
