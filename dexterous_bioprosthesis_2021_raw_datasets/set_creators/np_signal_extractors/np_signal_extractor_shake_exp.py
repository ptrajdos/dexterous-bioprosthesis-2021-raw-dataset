"""Module implementing shake expectation extraction.

Computes the mean absolute second difference of each channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorShakeExpectation(NPSignalExtractor):
    """Extractor that computes the mean absolute second difference."""

    def _transform(self, X):
        expe = np.abs(np.diff(X,n=2, axis=0))

        return np.mean(expe,axis=0)

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
