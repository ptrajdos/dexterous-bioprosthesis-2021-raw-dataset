"""Module implementing single-window signal extraction using milliseconds.

Extracts a single window of specified offset and length in milliseconds from each signal.
"""
from copy import deepcopy

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterWindowSingleMs(RawSignalsFilter):
    """Filter that extracts a single window from each signal using milliseconds."""

    def __init__(self, offset_ms: float, length_ms: float) -> None:
        """Cuts single window from given RawSignals using millisecond parameters.

        Arguments:
        ---------
        offset_ms:float offset from the beginning of the signal in milliseconds
        length_ms:float length of the window in milliseconds

        """
        super().__init__()
        self.offset_ms = offset_ms
        self.length_ms = length_ms

    def fit(self, raw_signals: RawSignals, y=None):
        """Does nothing
        """
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Extracts a window from each signal based on ms offset and length.
        """
        self._check_fitted()
        copied_signals = deepcopy(raw_signals)

        for signal in copied_signals:
            sample_rate = signal.get_sample_rate()
            start_sample = int(self.offset_ms * sample_rate / 1000.0)
            end_sample = int((self.offset_ms + self.length_ms) * sample_rate / 1000.0)
            signal.signal = signal.signal[start_sample:end_sample, :]

        return copied_signals
