from typing import Union, List, Tuple

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_single_if import (
    RawSignalsFilterWindowSingleIF,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_splitter.iraw_signals_splitter import (
    IRawSignalsSplitter,
)


class RawSignalsSplitterTemporal(IRawSignalsSplitter):
    def __init__(self, split_point):
        self.split_point = split_point

    def split_signals(
        self, raw_signals: RawSignals
    ) -> Union[List[RawSignals], Tuple[RawSignals]]:
        train_filter = RawSignalsFilterWindowSingleIF(offset=0, length=self.split_point)
        train_data = train_filter.fit_transform(raw_signals)

        test_filter = RawSignalsFilterWindowSingleIF(
            offset=self.split_point, length=None
        )
        test_data = test_filter.fit_transform(raw_signals)

        return train_data, test_data
