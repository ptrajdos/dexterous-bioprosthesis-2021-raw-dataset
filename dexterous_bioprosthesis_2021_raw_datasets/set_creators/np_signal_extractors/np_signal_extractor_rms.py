"""Module implementing Root Mean Square (RMS) extraction.

Computes the RMS value of each channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorRms(NPSignalExtractor):
    """Extractor that computes the Root Mean Square value."""

    def _transform(self, X):

        return np.sqrt(np.mean(X**2, axis=0))

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
