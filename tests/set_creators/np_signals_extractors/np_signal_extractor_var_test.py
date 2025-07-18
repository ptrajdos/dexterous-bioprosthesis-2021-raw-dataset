from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_var import NpSignalExtractorVar
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import NpSignalExtractorTest

import numpy as np

class NpSignalExtractorVarTest(NpSignalExtractorTest):


    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorVar(),
            NpSignalExtractorVar(sanitize_output=True,check_input=True, check_output=True),
        ]
    
