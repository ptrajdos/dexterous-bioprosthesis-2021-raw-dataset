"""Module implementing cubic soft-clipping distortion.

Applies a cubic polynomial clipping function calibrated to a target SNR.
"""
from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler import (
    RawSignalsSpoiler,
)
from dexterous_bioprosthesis_2021_raw_datasets.tools.numba_compat import jit


class RawSignalsSpoilerCubicClipper(RawSignalsSpoiler):
    """Spoiler that applies cubic soft-clipping distortion calibrated to a target SNR.

    The cubic soft clipper has a small-signal gain of 1.5, so it always
    distorts the signal: SNRs above roughly 11 dB cannot be reached. A warning
    is issued when the target SNR is missed. Negative SNR is clamped to 0.
    """

    def __init__(self, channels_spoiled_frac=0.1, snr=1, random_state=10,) -> None:
        super().__init__(channels_spoiled_frac, snr, random_state)

    @staticmethod
    def _distort(channel, t):
        return clipper(channel.astype(np.float64), t=float(t))

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
                    channel, self._distort, guesses=(0.05, 0.2, 0.5, 1.0, 2.0), target_snr=self._effective_snr
                )
                np_sig[:, ch_idx] = self._distort(channel, param)

        return copied_signals


@jit(nopython=True)
def clipper(s, t=0.666666, tol=1e-10):
    """Apply the cubic clipping function to a signal value."""
    s_a = np.abs(s)
    max_val = np.max(s_a)
    threshold = (t * max_val).item()
    out = np.zeros_like(s)

    if threshold < tol:
        return out

    A = 1.5
    B = -0.5 / (threshold**2)

    n_samples = s.shape[0]

    for sample_id in np.arange(n_samples):
        if s_a[sample_id] > threshold:
            out[sample_id] = np.sign(s[sample_id]) * threshold
        else:
            out[sample_id] = A * s[sample_id] + B * np.power(s[sample_id], 3)

    return out
