"""Module implementing row selection by slice for each signal.

Selects rows (samples) from each RawSignal using a slice object.
"""

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import \
    RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import \
    RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import \
    RawSignalsFilter


class RawSignalsFilterRowSlice(RawSignalsFilter):
    """Filter that selects signal rows by a slice object.

    Parameters
    ----------
    row_slice : slice
        A Python slice object to select rows from each signal.
    """

    def __init__(self, row_slice) -> None:
        super().__init__()
        self.row_slice = row_slice

    def fit(self, raw_signals: RawSignals, y=None):
        """Does nothing."""
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_fitted()
        filtered_signals = RawSignals()
        for raw_signal in raw_signals:
            new_signal = raw_signal[self.row_slice]
            filtered_signals.append(new_signal)

        return filtered_signals
