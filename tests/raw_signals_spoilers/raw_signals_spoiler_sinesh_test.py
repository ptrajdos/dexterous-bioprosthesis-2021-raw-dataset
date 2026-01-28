from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_sinesh import RawSignalsSpoilerSinesHarmonics
from tests.raw_signals_spoilers.raw_signals_spoiler_test import RawSignalsSpoilerTest


class RawSignalsSpoilerSinesHarmonicsTest(RawSignalsSpoilerTest):

    __test__ = True

    def get_spoilers(self):
        return [
            RawSignalsSpoilerSinesHarmonics(),
            RawSignalsSpoilerSinesHarmonics(channels_spoiled_frac=0),
            RawSignalsSpoilerSinesHarmonics(channels_spoiled_frac=1.0),
            RawSignalsSpoilerSinesHarmonics(channels_spoiled_frac=None),
        ]
    
    def get_spoiler_class(self):
        return RawSignalsSpoilerSinesHarmonics
    
    def is_test_snr(self):
        return True
    