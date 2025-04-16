from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_awl import NpSignalExtractorAWL
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mav import NpSignalExtractorMav
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import NpSignalExtractorTest
import numpy as np

class NpSignalExtractorAWLTest(NpSignalExtractorTest):


    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorAWL(), 
            NpSignalExtractorAWL(sanitize_output=True,check_input=True, check_output=True),
        ]
    

