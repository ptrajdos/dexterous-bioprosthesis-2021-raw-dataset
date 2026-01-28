from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_sine import RawSignalsSpoilerSine
from tests.raw_signals_spoilers.raw_signals_spoiler_test import RawSignalsSpoilerTest


class RawSignalsSpoilerSineTest(RawSignalsSpoilerTest):

    __test__ = True

    def get_spoilers(self):
        return [
            RawSignalsSpoilerSine(),
            RawSignalsSpoilerSine(channels_spoiled_frac=0),
            RawSignalsSpoilerSine(channels_spoiled_frac=1.0),
            RawSignalsSpoilerSine(channels_spoiled_frac=None),
        ]
    
    def get_spoiler_class(self):
        return RawSignalsSpoilerSine
    
    def is_test_snr(self):
        return True
    