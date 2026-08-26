"""Module implementing Discrete Wavelet Transform (DWT) based augmentation.

Decomposes signals using :func:`pywt.wavedec`, applies coefficient-level
transformations, and reconstructs via :func:`pywt.waverec`.
"""
from copy import deepcopy
from pywt import wavedec, waverec, dwt_max_level

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.wt_aug_base import (
    WTAugBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal


class WTAugDWT(WTAugBase):
    """Wavelet augmenter using the Discrete Wavelet Transform (DWT).

    Args:
        wavelets: List of wavelet names to choose from.
        max_decomposition_level: Maximum decomposition level.
        transformations: Coefficient transformations to apply.
        random_state: Random seed for reproducibility.
        mode: Signal extension mode for DWT (e.g. ``'symmetric'``).
    """

    def __init__(
        self,
        wavelets=None,
        max_decomposition_level: int = 3,
        transformations=None,
        random_state=10,
        mode="symmetric",
    ) -> None:
        super().__init__(
            wavelets,
            max_decomposition_level,
            transformations,
            random_state=random_state,
        )
        self.mode = mode

    def _wt_trans(self, raw_signal: RawSignal, wavelet, level: int) -> list:
        """Decompose a signal using DWT.

        Args:
            raw_signal: The signal to decompose.
            wavelet: Wavelet name.
            level: Decomposition level (clamped to max allowed).

        Returns:
            List of wavelet decomposition coefficients.
        """
        np_sig = raw_signal.to_numpy()
        max_level = dwt_max_level(len(np_sig), wavelet)
        t_level = min(level, max_level)
        decomp_list = wavedec(
            np_sig, wavelet=wavelet, level=t_level, axis=0, mode=self.mode
        )
        return decomp_list

    def _wt_itrans(self, raw_signal: RawSignal, wavelet, decomps: list) -> RawSignal:
        """Reconstruct a signal from DWT coefficients.

        Args:
            raw_signal: Original signal (used as template for metadata).
            wavelet: Wavelet name.
            decomps: Modified decomposition coefficients.

        Returns:
            Reconstructed :class:`RawSignal`.
        """
        new_signal = deepcopy(raw_signal)
        rec_sig = waverec(coeffs=decomps, wavelet=wavelet, axis=0, mode=self.mode)
        new_signal.signal = rec_sig
        return new_signal
