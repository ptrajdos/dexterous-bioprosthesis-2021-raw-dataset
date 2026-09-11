"""Module implementing signal filtering by object_class value.

Keeps only signals whose numeric class label is in a given set of values.
Compatible with int, float, and numpy scalar classes.
"""

import numbers

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import \
    RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import \
    RawSignalsFilter


class RawSignalsFilterClassValue(RawSignalsFilter):
    """Filter that keeps signals whose numeric object_class is in the allowed set.

    Works with int, float, numpy scalars, and numpy array classes.
    For numpy array classes, the signal is kept if all elements are
    in the allowed values set.
    """

    def __init__(self, allowed_values) -> None:
        super().__init__()
        self.allowed_values = set(allowed_values)

    def fit(self, raw_signals: RawSignals, y=None):
        """Does nothing."""
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_fitted()
        filtered_signals = RawSignals()
        for raw_signal in raw_signals:
            label = raw_signal.object_class
            if isinstance(label, np.ndarray):
                match = all(v in self.allowed_values for v in label.flat)
            elif isinstance(label, numbers.Number):
                match = label in self.allowed_values
            else:
                match = label in self.allowed_values

            if match:
                filtered_signals.append(raw_signal)

        return filtered_signals
