"""Module providing a dummy (identity) augmenter for raw signals.

The :class:`RawSignalsAugumenterDummy` returns deep copies of the input
signals without applying any transformation. It is useful as a baseline
or placeholder in augmentation pipelines.
"""
from copy import deepcopy

import numpy as np
from sklearn.exceptions import NotFittedError

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsAugumenterDummy(RawSignalsAugumenter):
    """Dummy augmenter that returns unmodified copies of the input signals.

    Acts as an identity transformation, producing deep copies of the
    original dataset without any signal modification.
    """

    def __init__(self) -> None:
        super().__init__()

    def _check_fittesness(self):
        """Raise :class:`NotFittedError` if the augmenter has not been fitted."""
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def fit(self, raw_signals: RawSignals, y=None):
        """Mark the augmenter as fitted.

        Args:
            raw_signals: Input dataset (not used).
            y: Ignored. Present for API compatibility.

        Returns:
            The fitted augmenter instance.
        """
        self._is_fitted = True
        return self

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        """Return a deep copy of the input signals without modification.

        Args:
            raw_signals: The dataset to copy.

        Returns:
            Deep copy of the input :class:`RawSignals`.
        """
        self._check_fittesness()
        return deepcopy(raw_signals)

    def fit_transform(self, raw_signals: RawSignals, y=None) -> RawSignals:
        """Fit and return an unmodified copy of the dataset.

        Args:
            raw_signals: The dataset to copy.
            y: Ignored. Present for API compatibility.

        Returns:
            Deep copy of the input :class:`RawSignals`.
        """
        self.fit(raw_signals, y)
        return self.transform(raw_signals)

    def sample(self, raw_signals: RawSignals, n_samples: int=1) -> RawSignals:
        """Randomly sample signals from the dataset without augmentation.

        Args:
            raw_signals: The dataset to sample from.
            n_samples: Number of samples to draw.

        Returns:
            A new :class:`RawSignals` with ``n_samples`` copied signals.
        """
        self._check_fittesness()
        n_signals = len(raw_signals)

        replace = n_samples > n_signals
        indices = np.random.choice(n_signals, size=n_samples, replace=replace)
        new_signals = raw_signals.initialize_empty()
        for idx in indices:
            new_signals += [deepcopy(raw_signals[idx])]

        return new_signals