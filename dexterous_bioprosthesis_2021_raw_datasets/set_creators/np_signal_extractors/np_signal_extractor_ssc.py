"""Module implementing Slope Sign Change (SSC) extraction.

Computes the fraction of samples, per channel, at which the slope changes sign.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorSsc(NPSignalExtractor):
    """Extractor that computes the Slope Sign Change rate."""

    def _transform(self, X):
        dx = np.diff(X, axis=0)
        return np.mean((dx[:-1] * dx[1:] < 0).astype(X.dtype), axis=0)

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
