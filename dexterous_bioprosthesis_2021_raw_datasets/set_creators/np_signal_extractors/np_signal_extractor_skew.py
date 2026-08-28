"""Module implementing skewness extraction.

Computes the skewness of each signal channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np
from scipy.stats import skew


class NpSignalExtractorSkew(NPSignalExtractor):
    """Extractor that computes the skewness of each signal channel."""

    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        return super().fit(X)

    def _transform(self, X):

        #FIXME Is there a better way to handle NaN/Inf values?
        skew_values = skew(X, nan_policy="propagate")
        skew_values = np.nan_to_num(skew_values, nan=0.0, posinf=0.0, neginf=0.0).astype(X.dtype)
        return skew_values
        

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
