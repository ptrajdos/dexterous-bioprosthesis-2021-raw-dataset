"""Module implementing additive sinusoidal interference.

Adds sine-wave interference calibrated to a target signal-to-noise ratio.
"""
from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler import (
    RawSignalsSpoiler,
)


class RawSignalsSpoilerSine(RawSignalsSpoiler):
    """Spoiler that adds sinusoidal interference calibrated to a target SNR.

    The frequency (frequency +- freq_deviation) and phase are drawn once per
    ``transform`` call and shared by all signals, like mains interference.
    """

    def __init__(
        self,
        channels_spoiled_frac=0.1,
        snr=1,
        random_state=10,
        frequency=50,
        freq_deviation=2,
    ) -> None:
        super().__init__(channels_spoiled_frac, snr, random_state)
        self.frequency = frequency
        self.freq_deviation = freq_deviation

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_is_fitted()
        copied_signals = deepcopy(raw_signals)

        used_frequency = (
            2 * self.freq_deviation * self._random_state.random(1)
            + self.frequency
            - self.freq_deviation
        )
        phase_shift = 4 * np.pi * self._random_state.random(1) - 2 * np.pi

        for signal in copied_signals:
            selected_channels_idxs = self._random_channel_selection(signal)
            np_sig = self._float_signal(signal)
            n_samples, n_channels = np_sig.shape

            x = np.arange(n_samples) / signal.get_sample_rate()
            noise_sig_np_col = np.reshape(
                np.sin(2 * np.pi * used_frequency * x + phase_shift), (n_samples, -1)
            )
            noise_sig_np = np.repeat(noise_sig_np_col, n_channels, axis=1)
            acc_noise_powers = self._channel_powers(noise_sig_np)

            desired_noise_powers = self._desired_channel_noise_powers(np_sig)

            alphas = np.sqrt(desired_noise_powers / acc_noise_powers)

            np_sig[:, selected_channels_idxs] += (
                alphas[selected_channels_idxs] * noise_sig_np[:, selected_channels_idxs]
            )

        return copied_signals
