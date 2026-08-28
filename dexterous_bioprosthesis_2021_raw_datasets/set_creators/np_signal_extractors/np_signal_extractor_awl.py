"""Module implementing Average Waveform Length (AWL) extraction.

Computes the average waveform length feature for each channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorAWL(NPSignalExtractor):
    """Extractor that computes the Average Waveform Length."""

    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        return super().fit(X)

    def _transform(self, X):
        differences = np.abs(np.diff(X, axis=0))

        awl = np.mean(differences, axis=0)

        return awl

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
