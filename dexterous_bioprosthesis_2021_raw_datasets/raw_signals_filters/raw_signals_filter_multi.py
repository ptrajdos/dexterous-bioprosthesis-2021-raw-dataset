"""Module implementing a parallel multi-filter combiner.

Applies multiple filters independently and concatenates their outputs
along the channel axis.
"""
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_pass import (
    RawSignalsFilterAllPass,
)


class RawSignalsFilterMulti(RawSignalsFilter):
    """Parallel multi-filter combiner that concatenates filter outputs."""
    def __init__(self, filter_list=[RawSignalsFilterAllPass()]) -> None:
        super().__init__()

        self.filter_list = filter_list

    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals) -> RawSignals:

        """Transform the given data."""
        self._check_fitted()

        pre_signals: RawSignals = raw_signals
        post_signals: RawSignals = None
        for filter in self.filter_list:
            # lazy fitting. Depends on filter order
            post_signals = filter.fit_transform(pre_signals)
            pre_signals = post_signals

        return post_signals
