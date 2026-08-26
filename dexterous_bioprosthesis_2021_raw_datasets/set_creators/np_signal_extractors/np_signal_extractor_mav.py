"""Module implementing Mean Absolute Value (MAV) extraction.

Computes the mean absolute value of each channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorMav(NPSignalExtractor):

    """Extractor that computes the Mean Absolute Value."""
    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        return super().fit(X)

    def _transform(self, X):

        return np.mean(np.abs(X), axis=0)

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
