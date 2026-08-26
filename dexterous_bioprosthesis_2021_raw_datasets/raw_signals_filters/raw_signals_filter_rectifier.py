"""Module implementing full-wave rectification.

Applies absolute value to all signal samples.
"""
from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterRectifier(RawSignalsFilter):

    """Filter that applies full-wave rectification to signals."""
    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_fitted()

        copied_signals = deepcopy(raw_signals)
        for r_signal in copied_signals:
            r_signal.signal = np.abs(r_signal.signal)

        return copied_signals
