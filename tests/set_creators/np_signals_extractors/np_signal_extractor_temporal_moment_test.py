from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_temporal_moment import NpSignalExtractorTemporalMoment
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import (
    NpSignalExtractorTest,
)


class NpSignalExtractorTemporalMomentTest(NpSignalExtractorTest):

    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorTemporalMoment(),
            NpSignalExtractorTemporalMoment(
                sanitize_output=True, check_input=True, check_output=True
            ),
            NpSignalExtractorTemporalMoment(order=0),
            NpSignalExtractorTemporalMoment(order=2, central=True),
            NpSignalExtractorTemporalMoment(order=3, central=False, proportional_time=False),
        ]
