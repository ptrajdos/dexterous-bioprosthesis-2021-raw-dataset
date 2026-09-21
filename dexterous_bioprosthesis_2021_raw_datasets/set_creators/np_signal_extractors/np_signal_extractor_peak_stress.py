"""Module implementing Root Mean Square (RMS) extraction.

Computes the RMS value of each channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorPeakStress(NPSignalExtractor):
    """Extractor that computes the Root Mean Square value."""

    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        return super().fit(X)

    def _transform(self, X):
        m2 = np.mean(X ** 2, axis=0)
        m4 = np.mean(X ** 4, axis=0)

        return np.divide(
            m4,
            m2,
            out=np.zeros_like(m4),
            where=m2 > 0,
        )

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
