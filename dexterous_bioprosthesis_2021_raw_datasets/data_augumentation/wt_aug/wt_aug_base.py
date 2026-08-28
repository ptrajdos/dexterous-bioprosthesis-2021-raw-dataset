"""Module providing the abstract base class for wavelet transform augmenters.

Defines :class:`WTAugBase`, which implements the common workflow of
decomposing signals via a wavelet transform, applying coefficient-level
transformations, and reconstructing the augmented signals.
"""
from __future__ import annotations

import abc
from typing import Optional

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_dummy import (
    DecompTransformationDummy,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class WTAugBase(RawSignalsAugumenter):
    """Abstract base class for wavelet transform-based augmenters.

    Subclasses must implement :meth:`_wt_trans` and :meth:`_wt_itrans` to
    define the forward and inverse wavelet transforms. The augmentation
    pipeline randomly selects a wavelet, decomposition level, and
    coefficient transformation for each signal.

    Args:
        wavelets: List of wavelet names to choose from. Defaults to
            ``['db4', 'db6']``.
        max_decomposition_level: Maximum wavelet decomposition level.
        transformations: List of :class:`IDecompTransformation` instances.
            Defaults to :class:`DecompTransformationDummy`.
        random_state: Random seed for reproducibility.

    """

    def __init__(
        self,
        wavelets: Optional[list] = None,
        max_decomposition_level: int = 3,
        transformations: Optional[list] = None,
        random_state=10,
    ) -> None:
        super().__init__()
        self.wavelets = wavelets
        self.max_decomposition_level = max_decomposition_level
        self.transformations = transformations
        self.random_state = random_state

    def _set_effective_wavelets(self) -> None:
        """Set the effective wavelet list, using defaults if not provided."""
        if self.wavelets is None:
            self._effective_wavelets = ["db4", "db6"]
        else:
            self._effective_wavelets = self.wavelets

    def _set_effective_transformations(self) -> None:
        """Set the effective transformations list and fit each one."""
        if self.transformations is None:
            self._effective_transformations: list = [DecompTransformationDummy()]
        else:
            self._effective_transformations: list = self.transformations

        for r_trans in self._effective_transformations:
            r_trans.fit()

    def fit(self, raw_signals: RawSignals):
        """Fit the augmenter by initialising wavelets, transformations, and random state.

        Args:
            raw_signals: The input dataset (used for API compatibility).

        Returns:
            The fitted augmenter instance.

        """
        self._set_effective_wavelets()
        self._set_effective_transformations()
        self._is_fitted = True
        self._random_state = check_random_state(self.random_state)
        return self

    def _check_fitted(self):
        """Raise :class:`NotFittedError` if the augmenter has not been fitted."""
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def _select_params(self, raw_signals: RawSignals) -> tuple:
        """Randomly select wavelet, level, and transformation per signal.

        Args:
            raw_signals: The dataset to select parameters for.

        Returns:
            Tuple of (wavelets, levels, transformations) arrays.

        """
        n_signals = len(raw_signals)
        sel_wavelets = self._random_state.choice(
            self._effective_wavelets, size=n_signals, replace=True
        )
        sel_levels = self._random_state.choice(
            np.arange(1, self.max_decomposition_level + 1), size=n_signals, replace=True
        )
        sel_transformations = self._random_state.choice(
            self._effective_transformations, size=n_signals, replace=True
        )

        return (sel_wavelets, sel_levels, sel_transformations)

    @abc.abstractmethod
    def _wt_trans(self, raw_signal: RawSignal, wavelet, level: int) -> list:
        """Transforms a signal using a kind of wavelet transform
        """

    @abc.abstractmethod
    def _wt_itrans(self, raw_signal: RawSignal, wavelet, decomps: list) -> RawSignal:
        """Inverse transformation
        """

    def _apply_transformation(self, trans, decomp) -> list:
        """Apply a decomposition transformation to wavelet coefficients."""
        return trans.transform(decomp)

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        """Transform signals via wavelet decomposition, modification, and reconstruction.

        Args:
            raw_signals: The dataset to augment.

        Returns:
            Augmented :class:`RawSignals`.

        """
        self._check_fitted()
        sel_wavelets, sel_levels, sel_transformations = self._select_params(raw_signals)
        new_signals = raw_signals.initialize_empty()

        for wav, lvl, trans, sig in zip(
            sel_wavelets, sel_levels, sel_transformations, raw_signals
        ):
            decomp = self._wt_trans(sig, wav, lvl)
            transformed = self._apply_transformation(trans, decomp)
            t_sig = self._wt_itrans(sig, wav, transformed)
            new_signals.append(t_sig)

        return new_signals

    def sample(self, raw_signals: RawSignals, n_samples: int = 1) -> RawSignals:
        """Randomly sample signals and augment them via wavelet transform.

        Args:
            raw_signals: The dataset to sample from.
            n_samples: Number of samples to produce.

        Returns:
            A new :class:`RawSignals` with ``n_samples`` augmented signals.

        """
        self._check_fitted()
        n_sigs = len(raw_signals)
        replace = n_samples > n_sigs
        indices = self._random_state.choice(n_sigs, n_samples, replace=replace)
        sampled_signals = raw_signals.initialize_empty()
        sampled_signals += raw_signals[indices]
        return self.transform(sampled_signals)

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        """Fit the augmenter and transform the dataset.

        Args:
            raw_signals: The dataset to augment.

        Returns:
            Augmented :class:`RawSignals`.

        """
        self.fit(raw_signals)
        return self.transform(raw_signals)
