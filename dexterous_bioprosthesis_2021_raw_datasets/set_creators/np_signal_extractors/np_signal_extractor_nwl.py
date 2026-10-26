"""Module implementing Normalised Waveform Length (NWL) extraction.

Computes the waveform length of each channel divided by the number of
sample differences.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorNWL(NPSignalExtractor):
    """Extractor that computes the Normalised Waveform Length."""

    def _transform(self, X):
        nwl = np.sum(np.abs(np.diff(X, axis=0)), axis=0) / (len(X) - 1)
        return nwl

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
