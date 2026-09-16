from typing import List, Tuple, Union
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_splitter.iraw_signals_splitter import IRawSignalsSplitter
from copy import deepcopy

class RawSignalsSplitterDummy(IRawSignalsSplitter):
    

    def split_signals(self, raw_signals: RawSignals) -> Union[List[RawSignals], Tuple[RawSignals]]:

        return (deepcopy(raw_signals),)