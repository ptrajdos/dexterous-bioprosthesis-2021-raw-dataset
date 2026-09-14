"""Module implementing variance extraction.

Computes the variance of each signal channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorQuantile(NPSignalExtractor):
    """Extractor that computes the variance of each signal channel."""

    def __init__(self,q=0.5 ,sanitize_output=False, check_input=False, check_output=False):
        super().__init__(sanitize_output=sanitize_output, check_input=check_input, check_output=check_output)
        self.q = q

    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        return super().fit(X)

    def _transform(self, X):

        return np.quantile(X,self.q, axis=0)

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
