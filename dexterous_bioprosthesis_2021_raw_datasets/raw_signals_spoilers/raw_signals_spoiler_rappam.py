import warnings
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler import (
    RawSignalsSpoiler,
)
from copy import deepcopy
import numpy as np
from scipy.optimize import bisect, broyden1, fsolve
import numba as nb


class RawSignalsSpoilerRappAM(RawSignalsSpoiler):

    def __init__(self, channels_spoiled_frac=0.1, snr=1, p=3) -> None:
        super().__init__(channels_spoiled_frac, snr)
        self.p = p

    def fit(self, raw_signals: RawSignals):

        if self.snr < 0:
            self._effective_snr = 0
            warnings.warn("SNR is negative. Setting SNR to 0")
        else:
            self._effective_snr = self.snr

        return self

    def _find_alpha(self, np_sig, ch_idx, guesses=(1.0,)) -> int:

        def channel_snr(alpha):
            res = (
                self._calculate_snrs(
                    np_sig[:, ch_idx],
                    self._rapp_amplifier(x=np_sig[:, ch_idx], sat=alpha)
                    - np_sig[:, ch_idx],
                )
                - self._effective_snr
            )
            return res

        best_alpha = 0
        snr_diff = np.inf
        for guess in guesses:

            tmp_alpha = fsolve(func=channel_snr, x0=(guess))[0]
            tmp_snr = channel_snr(tmp_alpha)

            tmp_snr_diff = np.abs(tmp_snr - self._effective_snr)
            if tmp_snr_diff < snr_diff:
                snr_diff = tmp_snr_diff
                best_alpha = tmp_alpha

        return best_alpha

    def transform(self, raw_signals: RawSignals):
        copied_signals = deepcopy(raw_signals)

        for signal in copied_signals:
            np_sig = signal.to_numpy()
            n_channels = np_sig.shape[1]
            selected_channels_idxs = self._random_channel_selection(signal)

            alphas = np.zeros(n_channels)

            for ch_idx in selected_channels_idxs:
                alphas[ch_idx] = self._find_alpha(np_sig, ch_idx)

                np_sig[:, ch_idx] = self._rapp_amplifier(
                    x=np_sig[:, ch_idx], sat=alphas[ch_idx]
                ).astype(np_sig.dtype)

        return copied_signals

    def _rapp_amplifier(self, x, gain=1.0, sat=1.5):
        """
        Rapp amplifier model (AM/AM only)

        x    : input signal
        gain : small-signal gain
        sat  : saturation level
        p    : smoothness factor (higher = harder clip)
        """
        x = gain * x
        d_sat = np.max(np.abs(x)) * sat
        d_sat+=1e-10
        return x / (1 + (np.abs(x) / d_sat) ** (2 * self.p)) ** (1 / (2 * self.p))
