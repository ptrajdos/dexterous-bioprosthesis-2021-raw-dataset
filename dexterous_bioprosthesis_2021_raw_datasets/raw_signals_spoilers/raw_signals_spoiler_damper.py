"""Module implementing signal damping distortion.

Attenuates signal amplitude by a factor calibrated to a target SNR.
"""
from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler import (
    RawSignalsSpoiler,
)


class RawSignalsSpoilerDamper(RawSignalsSpoiler):
    """Spoiler that attenuates selected channels.

    A channel x becomes (1 - a) * x, where a is chosen so that the removed part
    a * x has the power giving the target SNR. Negative SNR is clamped to 0
    (channel fully attenuated), since a > 1 would invert the signal.
    """

    def __init__(self, channels_spoiled_frac=0.1, snr=1, random_state=10,) -> None:
        super().__init__(channels_spoiled_frac, snr, random_state)

    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        self._effective_snr = self._effective_snr_or_clamp()
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_is_fitted()
        copied_signals = deepcopy(raw_signals)

        for signal in copied_signals:
            selected_channels_idxs = self._random_channel_selection(signal)
            np_sig = self._float_signal(signal)

            channel_powers = self._channel_powers(np_sig)
            desired_channel_noise_powers = self._desired_channel_noise_powers(
                np_sig, snr=self._effective_snr
            )

            eps = np.finfo(channel_powers.dtype).eps
            alphas = np.sqrt(desired_channel_noise_powers / (channel_powers + eps))

            np_sig[:, selected_channels_idxs] *= 1 - alphas[selected_channels_idxs]

        return copied_signals
