"""Module implementing per-column min-max standardisation.

Scales each channel independently to a specified range based on
per-column minimum and maximum values.
"""
from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterColumnMinMaxStandarizer(RawSignalsFilter):
    """Filter that applies per-column min-max standardisation."""

    def __init__(self, eps=1e-30, range_min=0, range_max=1) -> None:
        super().__init__()
        self.eps = eps
        self.range_min = range_min
        self.range_max = range_max

    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        np_data = raw_signals.to_numpy_concat()
        self._min = np.min(np_data, axis=(0,))
        self._max = np.max(np_data, axis=(0,))

        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_fitted()

        copied_signals = deepcopy(raw_signals)
        for r_signal in copied_signals:
            sig_dtype = r_signal.to_numpy().dtype
            r_signal.signal = (
                ((r_signal.signal - self._min) / (self._max - self._min + self.eps))
                * (self.range_max - self.range_min)
                + self.range_min
            ).astype(sig_dtype)

        return copied_signals
