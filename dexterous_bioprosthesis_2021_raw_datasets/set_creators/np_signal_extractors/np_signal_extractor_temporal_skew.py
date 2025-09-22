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
            X,
            3,
            NpSignalExtractorTemporalMoment._get_time_vector(X, self.proportional_time),
            True,
        )
        u2 = NpSignalExtractorTemporalMoment._calculate_moment(
            X,
            2,
            NpSignalExtractorTemporalMoment._get_time_vector(X, self.proportional_time),
            True,
        )
        skew = u3 / (u2**1.5)

        return skew

    def attribs_per_column(self):
        return 1
