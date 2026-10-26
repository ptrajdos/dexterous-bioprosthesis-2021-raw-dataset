"""Module implementing variance extraction.

Computes the variance of each signal channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorVar(NPSignalExtractor):
    """Extractor that computes the variance of each signal channel."""

    def _transform(self, X):

        return np.var(X, axis=0)

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
