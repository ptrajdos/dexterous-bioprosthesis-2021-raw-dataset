"""Module implementing channel selection by column name regex.

Selects a subset of signal channels whose names match a given regular expression.
"""

import re

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import \
    RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import \
    RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import \
    RawSignalsFilter


class RawSignalsFilterColumnNameRegex(RawSignalsFilter):
    """Filter that selects signal channels whose names match a regex pattern."""

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
            object_class = raw_signal.object_class
            timestamp = raw_signal.timestamp
            channel_names = raw_signal.channel_names

            matching_indices = [i for i, name in enumerate(channel_names) if compiled.search(name)]
            np_signal = np_signal[:, matching_indices]
            channel_names = [channel_names[idx] for idx in matching_indices]

            new_signal = RawSignal(signal=np_signal, object_class=object_class,
                                   channel_names=channel_names, timestamp=timestamp)
            filtered_signals.append(new_signal)

        return filtered_signals
