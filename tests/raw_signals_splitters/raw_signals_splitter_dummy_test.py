from typing import Dict

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_splitter.iraw_signals_splitter import IRawSignalsSplitter
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_splitter.raw_signals_splitter_dummy import \
    RawSignalsSplitterDummy
from tests.raw_signals_splitters.raw_signals_splitter_test import RawSignalsSplitterTest


class RawSignalsSplitterDummyTest(RawSignalsSplitterTest):
    __test__ = True
    def get_splitters(self) -> Dict[str, IRawSignalsSplitter]:
        return {"dummy": RawSignalsSplitterDummy()}

