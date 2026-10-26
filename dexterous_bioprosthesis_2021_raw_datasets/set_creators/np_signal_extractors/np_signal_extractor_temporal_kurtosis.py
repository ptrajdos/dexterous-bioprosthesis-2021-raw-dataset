"""Module implementing temporal kurtosis extraction.

Computes the kurtosis of the temporal energy distribution.
"""
import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_temporal_moment import (
    NpSignalExtractorTemporalMoment,
)


class NpSignalExtractorTemporalKurtosis(NPSignalExtractor):
    """Extractor that computes the temporal kurtosis of signal energy."""

    def __init__(self, sanitize_output=False, check_input=False, check_output=False, proportional_time=True):
        super().__init__(sanitize_output=sanitize_output, check_input=check_input, check_output=check_output)
        self.proportional_time = proportional_time

    def _transform(self, X):
        X_norm = NpSignalExtractorTemporalMoment._get_X_normalized_columns(X)
        time_vector = NpSignalExtractorTemporalMoment._get_time_vector(
            X, self.proportional_time, self._get_fs()
        )
        u4 = NpSignalExtractorTemporalMoment._calculate_moment(
            X=X_norm, order=4, time_vector=time_vector, central=True
        )
        u2 = NpSignalExtractorTemporalMoment._calculate_moment(
            X=X_norm, order=2, time_vector=time_vector, central=True
        )
        kurtosis = np.divide(
            u4,
            u2**2,
            out=np.full_like(u4, 3.0),  # excess kurtosis=0 where u2==0
            where=u2 > 0
        ) - 3.0  # subtract 3 for excess kurtosis

        return kurtosis

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
