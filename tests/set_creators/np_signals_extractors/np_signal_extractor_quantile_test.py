from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mean import (
    NpSignalExtractorMean,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_quantile import NpSignalExtractorQuantile
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import (
    NpSignalExtractorTest,
)


class NpSignalExtractorQuantileTest(NpSignalExtractorTest):

    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorQuantile(),
            NpSignalExtractorQuantile(q=0.1),
            NpSignalExtractorQuantile(q=0.9),
            NpSignalExtractorQuantile(
                sanitize_output=True, check_input=True, check_output=True
            ),
        ]
