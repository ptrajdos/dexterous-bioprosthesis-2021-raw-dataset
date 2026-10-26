"""Module implementing quantile extraction.

Computes the q-th quantile of each signal channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorQuantile(NPSignalExtractor):
    """Extractor that computes the q-th quantile of each signal channel."""

    def __init__(self,q=0.5 ,sanitize_output=False, check_input=False, check_output=False):
        super().__init__(sanitize_output=sanitize_output, check_input=check_input, check_output=check_output)
        self.q = q

    def _transform(self, X):

        return np.quantile(X,self.q, axis=0)

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1

    def get_name(self) -> str:
        """Return the base name of the extracted attribute."""
        return "quantile_{:g}".format(self.q)
