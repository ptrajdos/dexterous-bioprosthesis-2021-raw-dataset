"""Module implementing signal filtering by object_class using regex.

Keeps only signals whose string class label matches a given regular expression.
"""

import re

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import \
    RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import \
    RawSignalsFilter


class RawSignalsFilterClassRegex(RawSignalsFilter):
    """Filter that keeps signals whose object_class matches a regex pattern.

    The object_class is converted to string before matching.
    For numpy array classes, each element is converted to string and
    the signal is kept if any element matches.
    """

    def __init__(self, pattern) -> None:
        super().__init__()
        self.pattern = pattern

    def fit(self, raw_signals: RawSignals, y=None):
        """Does nothing."""
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_fitted()
        compiled = re.compile(self.pattern)
        filtered_signals = RawSignals()
        for raw_signal in raw_signals:
            label = raw_signal.object_class
            if isinstance(label, np.ndarray):
                match = any(compiled.search(str(v)) for v in label.flat)
            else:
                match = compiled.search(str(label)) is not None

            if match:
                filtered_signals.append(raw_signal)

        return filtered_signals
