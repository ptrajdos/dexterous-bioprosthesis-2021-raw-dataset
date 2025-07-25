import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ssc import NpSignalExtractorSsc
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import NpSignalExtractorTest


class NpSignalExtractorSscTest(NpSignalExtractorTest):


    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorSsc(),
            NpSignalExtractorSsc(sanitize_output=True, check_input=True, check_output=True),
        ]