from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_gauss import RawSignalsSpoilerGauss
from tests.raw_signals_spoilers.raw_signals_spoiler_test import RawSignalsSpoilerTest

class RawSignalsSpoilerSineTest(RawSignalsSpoilerTest):

    __test__ = True

    def get_spoilers(self):
        return [
            RawSignalsSpoilerGauss(),
            RawSignalsSpoilerGauss(channels_spoiled_frac=0),
        ]
    
    def get_spoiler_class(self):
        return RawSignalsSpoilerGauss
    
    def is_test_snr(self):
        return True
    