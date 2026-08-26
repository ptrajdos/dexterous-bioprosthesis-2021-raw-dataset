"""Module implementing Stationary Wavelet Transform (SWT) based augmentation.

Decomposes signals using :func:`pywt.swt`, applies coefficient-level
transformations, and reconstructs via :func:`pywt.iswt`.
"""
from copy import deepcopy
from pywt import swt, iswt, swt_max_level

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.wt_aug_base import (
    WTAugBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal


class WTAugSWT(WTAugBase):
    """Wavelet augmenter using the Stationary Wavelet Transform (SWT).

    Unlike DWT, SWT is translation-invariant, which can produce
    augmentations that better preserve temporal structure.
    """

    def _wt_trans(self, raw_signal: RawSignal, wavelet, level: int) -> list:
        """Decompose a signal using SWT.

        Args:
            raw_signal: The signal to decompose.
            wavelet: Wavelet name.
            level: Decomposition level (clamped to max allowed).

        Returns:
            List of wavelet decomposition coefficients.
        """
        np_sig = raw_signal.to_numpy()
        max_lvl = swt_max_level(len(np_sig))
        t_level = min(level, max_lvl)
        decomp_list = swt(
            raw_signal.to_numpy(),
            wavelet=wavelet,
            level=t_level,
            axis=0,
            norm=True,
            trim_approx=True,
        )
        return decomp_list

    def _wt_itrans(self, raw_signal: RawSignal, wavelet, decomps: list) -> RawSignal:
        """Reconstruct a signal from SWT coefficients.

        Args:
            raw_signal: Original signal (used as template for metadata).
            wavelet: Wavelet name.
            decomps: Modified decomposition coefficients.

        Returns:
            Reconstructed :class:`RawSignal`.
        """
        new_signal = deepcopy(raw_signal)
        rec_sig = iswt(coeffs=decomps, wavelet=wavelet, axis=0, norm=True)
        new_signal.signal = rec_sig
        return new_signal
