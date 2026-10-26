"""Module providing the base spoiler with SNR-based noise calibration.

Computes channel powers and calibrates noise levels to achieve a
target signal-to-noise ratio.
"""
import warnings

import numpy as np
from scipy.optimize import fsolve
from sklearn.utils import check_random_state
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_interface import (
    RawSignalsSpoilerInterface,
)


class RawSignalsSpoiler(RawSignalsSpoilerInterface):
    """Base spoiler class providing SNR-based noise calibration and channel selection."""

    def __init__(self, channels_spoiled_frac=0.1, snr=1, random_state=10) -> None:
        """Raw SignalsSpoiler -- introduces noises and other disturbances to signals

        Arguments:
        ---------
        channels_spoiled_frac -- fraction of channels to be contaminated
        snr - signal to noise ratio of the contaminations

        """
        super().__init__()
        self.channels_spoiled_frac = channels_spoiled_frac
        self.snr = snr
        self.random_state = random_state

    def _random_channel_selection(self, raw_signal: RawSignal) -> list[int]:
        """Return list of randomly selected channels

        Arguments:
        ---------
        raw_signals -- RawSignals object

        Returns:
        iterable of selected channel indices

        """
        if self.channels_spoiled_frac == 0:
            # Explicit zero is passed
            return []

        n_samples, n_channels = raw_signal.to_numpy().shape
        if self.channels_spoiled_frac is None:
            # Random number of channels from 1 to n_channels (inclusive)
            n_sel_channels = self._random_state.randint(1, n_channels + 1)
        else:
            n_sel_channels = int(
                np.min([np.ceil(self.channels_spoiled_frac * n_channels), n_channels])
            )

        sel_channel_indices = self._random_state.choice(
            [*range(n_channels)], size=n_sel_channels, replace=False
        )

        return list(sel_channel_indices)

    def _channel_powers(self, raw_signal_np) -> np.ndarray:
        powers = np.mean(np.power(raw_signal_np, 2), axis=0)
        return powers

    def _desired_channel_noise_powers(self, raw_signal_np, snr=None) -> np.ndarray:
        """Noise powers giving the target SNR (self.snr unless snr is given)."""
        if snr is None:
            snr = self.snr
        channel_powers = self._channel_powers(raw_signal_np)
        desired_powers = channel_powers * np.power(10.0, -snr / 10.0)

        return desired_powers

    @staticmethod
    def _float_signal(raw_signal: RawSignal) -> np.ndarray:
        """Return the signal array, converted in place to float64 if it is not floating.

        Spoilers add or multiply floats into the array, which fails (or truncates)
        for integer signals.
        """
        if not np.issubdtype(raw_signal.signal.dtype, np.floating):
            raw_signal.signal = raw_signal.signal.astype(np.float64)
        return raw_signal.signal

    def _effective_snr_or_clamp(self):
        """Return self.snr, or 0 (with a warning) if it is negative."""
        if self.snr < 0:
            warnings.warn("SNR is negative. Setting SNR to 0")
            return 0
        return self.snr

    def _find_param_for_snr(self, channel, distort, guesses, target_snr, tol_db=0.5):
        """Find the distortion parameter for which the distorted channel has the target SNR.

        Arguments:
        ---------
        channel -- 1D clean signal
        distort -- function (channel, param) -> distorted channel
        guesses -- starting points for the solver; the best solution is kept
        target_snr -- desired SNR in dB
        tol_db -- warn if the best solution misses the target by more than this

        Returns:
        -------
        best parameter found

        """

        def snr_error(param):
            param = np.atleast_1d(param)[0]
            return (
                self._calculate_snrs(channel, distort(channel, param) - channel)
                - target_snr
            )

        best_param = guesses[0]
        best_diff = np.inf
        for guess in guesses:
            with warnings.catch_warnings():
                # Convergence is checked below; fsolve's own warnings are redundant.
                warnings.simplefilter("ignore", RuntimeWarning)
                param = fsolve(func=snr_error, x0=guess)[0]
            diff = np.abs(snr_error(param))
            if diff < best_diff:
                best_diff = diff
                best_param = param

        if best_diff > tol_db:
            warnings.warn(
                "{}: target SNR {} dB not reached (missed by {:.2f} dB).".format(
                    type(self).__name__, target_snr, float(best_diff)
                )
            )

        return best_param

    def _calculate_snrs(self, raw_signal_np, noise_signal_np) -> np.ndarray:
        signal_powers = np.mean(np.power(raw_signal_np, 2), axis=0)
        noise_powers = np.mean(np.power(noise_signal_np, 2), axis=0)

        eps_floor = 1e-20
        eps = np.finfo(signal_powers.dtype).eps * np.max(
            (
                np.max(signal_powers),
                np.max(noise_powers),
                eps_floor,
            )
        )

        snrs = 10.0 * np.log10((signal_powers + eps) / (noise_powers + eps))

        return snrs

    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        self._random_state = check_random_state(self.random_state)
        # Does nothing
        return super().fit(raw_signals, y)
