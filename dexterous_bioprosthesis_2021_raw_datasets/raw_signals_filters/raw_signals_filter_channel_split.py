"""Module implementing channel splitting.

Splits each multi-channel signal into multiple single-channel signals.
"""

import warnings

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import \
    RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import \
    RawSignalsFilter
from dexterous_bioprosthesis_2021_raw_datasets.tools.warnings import \
    ChannelNamesDeprecation


class RawSignalsFilterChannelSplit(RawSignalsFilter):
    """Filter that splits each signal into single-channel signals.

    A signal with ``n`` channels produces ``n`` new signals, one per channel,
    in channel order. Channel names, class label, timestamp and sample rate
    are preserved.

    Note: the resulting signals have different channel names, so the
    returned RawSignals object mixes channel names. The related
    ChannelNamesDeprecation warning is suppressed during the split.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, raw_signals: RawSignals, y=None):
        """Does nothing.
        """
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Split every signal into single-channel signals."""
        self._check_fitted()
        new_signals = raw_signals.initialize_empty()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ChannelNamesDeprecation)
            for raw_signal in raw_signals:
                n_channels = raw_signal.signal.shape[1]
                for ch_idx in range(n_channels):
                    channel_signal = raw_signal[:, ch_idx:ch_idx + 1]
                    channel_signal.signal = channel_signal.signal.copy()
                    new_signals.append(channel_signal)

        return new_signals
