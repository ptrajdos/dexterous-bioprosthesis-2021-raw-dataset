"""Module implementing column-to-class extraction by column name regex.

Selects a subset of signal columns whose names match a given regular expression
and assigns their values as the new class label (numpy array) for each signal.
"""

import re

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import \
    RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import \
    RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import \
    RawSignalsFilter


class RawSignalsFilterColumnToClassRegex(RawSignalsFilter):
    """Filter that extracts signal columns by name regex and sets them as class labels (array)."""

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
            np_signal = raw_signal.signal
            timestamp = raw_signal.timestamp
            channel_names = raw_signal.channel_names

            matching_indices = [i for i, name in enumerate(channel_names) if compiled.search(name)]
            remaining_indices = [i for i in range(len(channel_names)) if i not in matching_indices]

            class_columns = np_signal[:, matching_indices]
            new_signal = np_signal[:, remaining_indices]
            new_channel_names = [channel_names[i] for i in remaining_indices]

            new_raw_signal = RawSignal(signal=new_signal,
                                       object_class=class_columns,
                                       channel_names=new_channel_names,
                                       timestamp=timestamp)
            filtered_signals.append(new_raw_signal)

        return filtered_signals
