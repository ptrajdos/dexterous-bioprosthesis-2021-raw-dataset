"""Module implementing hyperbolic tangent soft-clipping distortion.

Applies a tanh-based clipping function calibrated to a target SNR.
"""
from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler import (
    RawSignalsSpoiler,
)


class RawSignalsSpoilerTanhClipper(RawSignalsSpoiler):
    """Spoiler that applies tanh-based soft-clipping distortion.

    A channel x with peak amplitude m becomes (m / a) * tanh(a * x / m).
    The small-signal gain is 1 and the output saturates at m / a, so the
    distortion does not depend on the signal's amplitude; a -> 0 leaves the
    signal unchanged. Negative SNR is clamped to 0.
    """

    def __init__(
        self,
        channels_spoiled_frac=0.1,
        snr=1,
        random_state=10,
    ) -> None:
        super().__init__(channels_spoiled_frac, snr, random_state=random_state)

    @staticmethod
    def _distort(channel, a):
        a = np.abs(a)
        peak = np.max(np.abs(channel))
        if peak == 0 or a < 1e-12:
            return channel.copy()
        return (peak / a) * np.tanh(a * channel / peak)

    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        self._effective_snr = self._effective_snr_or_clamp()
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_is_fitted()
        copied_signals = deepcopy(raw_signals)

        for signal in copied_signals:
            np_sig = self._float_signal(signal)
            selected_channels_idxs = self._random_channel_selection(signal)

            for ch_idx in selected_channels_idxs:
                channel = np_sig[:, ch_idx].copy()
                param = self._find_param_for_snr(
                    channel, self._distort, guesses=(0.5, 1.0, 2.0, 5.0), target_snr=self._effective_snr
                )
                np_sig[:, ch_idx] = self._distort(channel, param)

        return copied_signals
