"""Module implementing a Modified Moving Average Value (MAV2) window filter.

Applies a weighted moving average with trapezoidal window weights.
"""
import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window import (
    RawSignalsFilterWindowFilter,
)


class RawSignalsFilterMAV2WindowFilter(RawSignalsFilterWindowFilter):
    """Modified Moving Average (MAV2) window filter with trapezoidal weights."""

    def __init__(self, window_length: int = 100) -> None:
        super().__init__(window_length)

    def channel_transform(self, data):
        """Apply the window transformation to a single channel."""
        N = self.window_length
        M = N // 2
        out = np.zeros(data.shape[0])
        for i in range(M, out.shape[0] - M):
            out[i] = np.sum(np.abs(data[i - M : i + M])) / N
        return out
