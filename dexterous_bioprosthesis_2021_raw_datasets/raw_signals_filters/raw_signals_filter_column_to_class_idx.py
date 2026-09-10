"""Module implementing column-to-class extraction by index.

Selects a subset of signal columns by index and assigns their values
as the new class label (numpy array) for each signal.
"""

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import \
    RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import \
    RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import \
    RawSignalsFilter


class RawSignalsFilterColumnToClassIdx(RawSignalsFilter):
    """Filter that extracts signal columns by index and sets them as class labels (array)."""

    def __init__(self, indices_list) -> None:
        super().__init__()
        self.indices_list = indices_list

    def fit(self, raw_signals: RawSignals, y=None):
        """Does nothing."""
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_fitted()
        filtered_signals = RawSignals()
        for raw_signal in raw_signals:
            np_signal = raw_signal.signal
            timestamp = raw_signal.timestamp
            channel_names = raw_signal.channel_names

            class_columns = np_signal[:, self.indices_list]
            remaining_mask = np.ones(np_signal.shape[1], dtype=bool)
            remaining_mask[self.indices_list] = False
            new_signal = np_signal[:, remaining_mask]
            new_channel_names = [channel_names[i] for i in range(len(channel_names)) if remaining_mask[i]]

            new_raw_signal = RawSignal(signal=new_signal,
                                       object_class=class_columns,
                                       channel_names=new_channel_names,
                                       timestamp=timestamp)
            filtered_signals.append(new_raw_signal)

        return filtered_signals
