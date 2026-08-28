"""Module implementing a Modified Moving Average Value (MAV3) window filter.

Applies a weighted moving average with a custom window shape.
"""
import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window import (
    RawSignalsFilterWindowFilter,
)


class RawSignalsFilterMAV3WindowFilter(RawSignalsFilterWindowFilter):
    """Modified Moving Average (MAV3) window filter with custom window weights."""

    def __init__(self, window_length: int = 100) -> None:
        super().__init__(window_length)

    def channel_transform(self, data):
        """Apply the window transformation to a single channel."""
        N = self.window_length
        M = N // 2
        out = np.zeros(data.shape[0])
        mask = np.ones(N)
        for i in range(len(mask)):
            if i < 0.25 * N or i > 0.75 * N:
                mask[i] = 0.5
        for i in range(M, out.shape[0] - M):
            out[i] = np.sum(np.multiply(np.abs(data[i - M : i + M]), mask)) / N
        return out
