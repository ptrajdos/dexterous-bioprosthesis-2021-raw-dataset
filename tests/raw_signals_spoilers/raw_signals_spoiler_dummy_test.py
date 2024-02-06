
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_dummy import RawSignalsSpoilerDummy
from tests.raw_signals_spoilers.raw_signals_spoiler_test import RawSignalsSpoilerTest


class RawSignalsSpoilerDummyTest(RawSignalsSpoilerTest):

    __test__ = True

    def get_spoilers(self):
        return [
            RawSignalsSpoilerDummy()
        ]
    
    def get_spoiler_class(self):
        return RawSignalsSpoilerDummy