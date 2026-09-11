"""Module implementing row range selection for each signal.

Selects a range of rows (samples) from each RawSignal in a RawSignals collection.
"""

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import \
    RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import \
    RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import \
    RawSignalsFilter


class RawSignalsFilterRowRange(RawSignalsFilter):
    """Filter that selects a row range from each signal.

    Parameters
    ----------
    start : int or None
        Start index of the row range (inclusive). None means from the beginning.
    stop : int or None
        Stop index of the row range (exclusive). None means to the end.
    step : int or None
        Step size. None means 1.
    """

    def __init__(self, start=None, stop=None, step=None) -> None:
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def fit(self, raw_signals: RawSignals, y=None):
        """Does nothing."""
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_fitted()
        row_slice = slice(self.start, self.stop, self.step)
        filtered_signals = RawSignals()
        for raw_signal in raw_signals:
            new_signal = raw_signal[row_slice]
            filtered_signals.append(new_signal)

        return filtered_signals
