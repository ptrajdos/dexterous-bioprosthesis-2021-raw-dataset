from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_deltarms import NpSignalExtractorDeltaRms
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import NpSignalExtractorTest


class NpSignalExtractorRmsTest(NpSignalExtractorTest):


    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorDeltaRms(window_length_ms=50, offset_ms=10),
            NpSignalExtractorDeltaRms(window_length_ms=50, offset_ms=10, sanitize_output=True,check_input=True, check_output=True),
        ]
