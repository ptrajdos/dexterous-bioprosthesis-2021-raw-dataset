"""Module implementing Slope Sign Change (SSC) extraction.

Counts the number of slope sign changes in each channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorSsc(NPSignalExtractor):
    """Extractor that computes the Slope Sign Change count."""

    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        return super().fit(X)

    def _transform(self, X):
        return np.mean(
            ((-np.diff(X, axis=0, prepend=1)[1:-1] * np.diff(X, axis=0)[1:]) > 0).astype(X.dtype),
            axis=0,
        )

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
