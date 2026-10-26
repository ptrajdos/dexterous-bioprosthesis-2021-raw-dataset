"""Module implementing Rapp model amplifier distortion.

Applies a Rapp-model AM/AM nonlinear distortion calibrated to a target SNR.
"""
from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler import (
    RawSignalsSpoiler,
)


class RawSignalsSpoilerRappAM(RawSignalsSpoiler):
    """Spoiler that applies Rapp-model AM/AM nonlinear distortion.

    Negative SNR is clamped to 0.
    """

    def __init__(self, channels_spoiled_frac=0.1, snr=1, random_state=10, p=3) -> None:
        super().__init__(channels_spoiled_frac, snr, random_state)
        self.p = p

    def _distort(self, channel, sat):
        return self._rapp_amplifier(x=channel, sat=sat)

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
                    channel, self._distort, guesses=(1.0,), target_snr=self._effective_snr
                )
                np_sig[:, ch_idx] = self._distort(channel, param)

        return copied_signals

    def _rapp_amplifier(self, x, gain=1.0, sat=1.5):
        """Rapp amplifier model (AM/AM only)

        x    : input signal
        gain : small-signal gain
        sat  : saturation level, as a fraction of the peak amplitude
        p    : smoothness factor (higher = harder clip)
        """
        x = gain * x
        d_sat = np.max(np.abs(x)) * sat
        d_sat += 1e-10
        return x / (1 + (np.abs(x) / d_sat) ** (2 * self.p)) ** (1 / (2 * self.p))
