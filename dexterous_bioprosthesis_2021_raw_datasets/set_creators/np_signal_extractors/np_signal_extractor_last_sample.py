"""Module implementing last sample extraction.

Extracts the last sample of each signal channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorLastSample(NPSignalExtractor):
    """Extractor that extracts the last sample of each signal channel."""

    def _transform(self, X):

        return np.copy(X[-1])

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
