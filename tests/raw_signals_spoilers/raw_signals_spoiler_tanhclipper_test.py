import unittest

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_tanhclipper import RawSignalsSpoilerTanhClipper
from tests.raw_signals_spoilers.raw_signals_spoiler_test import RawSignalsSpoilerTest

import numpy as np

class RawSignalsSpoilerTanhClipperTest(RawSignalsSpoilerTest):

    __test__ = True

    def get_spoilers(self):
        return [
            RawSignalsSpoilerTanhClipper(),
            RawSignalsSpoilerTanhClipper(channels_spoiled_frac=0),
            RawSignalsSpoilerTanhClipper(channels_spoiled_frac=1.0),
            RawSignalsSpoilerTanhClipper(channels_spoiled_frac=None),
        ]

    def get_spoiler_class(self):
        return RawSignalsSpoilerTanhClipper
    
    def is_test_snr(self):
        return True
    
    def get_snrs(self):
        return [0,1,2,3,4,5]

if __name__ == '__main__':
    unittest.main()