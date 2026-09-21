"""Module implementing Root Mean Square (RMS) extraction.

Computes the RMS value of each channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorPeakStress2(NPSignalExtractor):
    """Extractor that computes the Root Mean Square value."""

    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        return super().fit(X)

    def _transform(self, X):
        m0 = np.mean(X**2, axis=0)

        dx = np.diff(X, axis=0)
        m2 = np.mean(dx**2, axis=0)

        d2x = np.diff(X, n=2, axis=0)
        m4 = np.mean(d2x**2, axis=0)

        denominator = m2 * m0

        return np.divide(
            m4,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
