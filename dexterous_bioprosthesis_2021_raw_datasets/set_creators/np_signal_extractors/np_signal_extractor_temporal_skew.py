import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_temporal_moment import (
    NpSignalExtractorTemporalMoment,
)


class NpSignalExtractorTemporalSkew(NPSignalExtractor):

    def __init__(self, sanitize_output=False, check_input=False, check_output=False, proportional_time=True):
        super().__init__(sanitize_output=sanitize_output, check_input=check_input, check_output=check_output)
        self.proportional_time = proportional_time

    def _transform(self, X):
        u3 = NpSignalExtractorTemporalMoment._calculate_moment(
            X=NpSignalExtractorTemporalMoment._get_X_normalized_columns(X),
            order=3,
            time_vector=NpSignalExtractorTemporalMoment._get_time_vector(X, self.proportional_time),
            central=True,
        )
        u2 = NpSignalExtractorTemporalMoment._calculate_moment(
            X=NpSignalExtractorTemporalMoment._get_X_normalized_columns(X),
            order= 2,
            time_vector=NpSignalExtractorTemporalMoment._get_time_vector(X, self.proportional_time),
            central=True,
        )
        skew = np.divide(
            u3,
            u2**1.5,
            out=np.zeros_like(u3),   # set skew=0 where u2==0
            where=u2 > 0             # only divide where u2 > 0
        )

        return skew

    def attribs_per_column(self):
        return 1
