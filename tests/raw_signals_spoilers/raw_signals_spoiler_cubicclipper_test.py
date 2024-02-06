from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_cubicclipper import RawSignalsSpoilerCubicClipper
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_tanhclipper import RawSignalsSpoilerTanhClipper
from tests.raw_signals_spoilers.raw_signals_spoiler_test import RawSignalsSpoilerTest

import numpy as np

class RawSignalsSpoilerCubicClipperTest(RawSignalsSpoilerTest):

    __test__ = True

    def get_spoilers(self):
        return [
            RawSignalsSpoilerCubicClipper(),
            RawSignalsSpoilerCubicClipper(channels_spoiled_frac=0),
        ]

    def get_spoiler_class(self):
        return RawSignalsSpoilerCubicClipper
    
    def is_test_snr(self):
        return True
    
    def get_snrs(self):
        return [0,1,2,3,4]