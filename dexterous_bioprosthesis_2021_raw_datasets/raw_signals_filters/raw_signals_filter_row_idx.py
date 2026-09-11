"""Module implementing row selection by index for each signal.

Selects specific rows (samples) by index from each RawSignal in a RawSignals collection.
"""

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import \
    RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import \
    RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import \
    RawSignalsFilter


class RawSignalsFilterRowIdx(RawSignalsFilter):
    """Filter that selects signal rows by index.

    Parameters
    ----------
    indices_list : list of int
        Row indices to select from each signal.
    """

    def __init__(self, indices_list) -> None:
        super().__init__()
        self.indices_list = indices_list

    def fit(self, raw_signals: RawSignals, y=None):
        """Does nothing."""
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_fitted()
        filtered_signals = RawSignals()
        for raw_signal in raw_signals:
            new_signal = raw_signal[self.indices_list]
            filtered_signals.append(new_signal)

        return filtered_signals
