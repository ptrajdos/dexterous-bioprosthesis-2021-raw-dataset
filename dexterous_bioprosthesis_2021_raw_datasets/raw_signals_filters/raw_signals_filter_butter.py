"""Module implementing a Butterworth filter for raw signals.

Applies a Butterworth IIR filter to each channel of each signal
using :func:`scipy.signal.sosfiltfilt`.
"""
from copy import deepcopy

from scipy import signal

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterButter(RawSignalsFilter):

    """Filter that applies a Butterworth IIR filter to each signal channel."""
    def __init__(self, low_freq=48, high_freq=52, order=4, btype="bandstop") -> None:
        super().__init__()
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        self.btype = btype

    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        """Transform the given data."""
        self._check_fitted()

        nyq = raw_signals.sample_rate * 0.5
        low = self.low_freq / nyq
        high = self.high_freq / nyq

        copied_signals = deepcopy(raw_signals)

        for raw_signal in copied_signals:
            sig_np = raw_signal.signal
            n_ch = sig_np.shape[1]

            for ch_idx in range(n_ch):
                b, a = signal.butter(self.order, [low, high], btype=self.btype)
                sig_np[:, ch_idx] = signal.filtfilt(b, a, sig_np[:, ch_idx])

        return copied_signals
