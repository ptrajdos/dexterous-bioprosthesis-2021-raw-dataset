"""Module implementing peak stress extraction.

Computes mean(x**4) / mean(x**2) for each channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorPeakStress(NPSignalExtractor):
    """Extractor that computes the peak stress, mean(x**4) / mean(x**2)."""

    def _transform(self, X):
        m2 = np.mean(X ** 2, axis=0)
        m4 = np.mean(X ** 4, axis=0)

        return np.divide(
            m4,
            m2,
            out=np.zeros_like(m4),
            where=m2 > 0,
        )

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
