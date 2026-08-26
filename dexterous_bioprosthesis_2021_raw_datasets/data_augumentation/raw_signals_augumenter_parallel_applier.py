"""Module implementing parallel (fan-out) augmentation applier.

Applies multiple augmenters independently to the same input dataset and
concatenates all results, effectively running augmenters in parallel.
"""
import numpy as np
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_invert_polarity import (
    RawSignalsAugumenterInvertPolarity,
)
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_white_noise import (
    RawSignalsAugumenterWhiteNoise,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsAugumenterParallelApplier(RawSignalsAugumenter):
    """Augmenter that applies multiple augmenters in parallel (fan-out).

    Each augmenter in the list independently transforms the full input
    dataset, and all outputs are concatenated into a single result.

    Args:
        augumenter_list: List of :class:`RawSignalsAugumenter` instances to
            apply. If ``None``, a default list is used.
        append_original: Whether to include original signals in the output.
    """

    def __init__(self, augumenter_list=None, append_original=True) -> None:
        super().__init__()

        self.augumenter_list = augumenter_list
        self.append_original = append_original

    def _prepare_effective_augumenter_list(self):
        """Initialise the effective augmenter list, using defaults if needed."""
        if self.augumenter_list is None or len(self.augumenter_list) == 0:
            self._augumenter_list = [
                RawSignalsAugumenterInvertPolarity(append_original=False),
                RawSignalsAugumenterWhiteNoise(
                    noise_perc_min=0.2, n_repeats=2, append_original=False
                ),
            ]
        else:
            self._augumenter_list = self.augumenter_list

    def _check_fitted(self):
        """Raise :class:`NotFittedError` if the applier has not been fitted."""
        if not hasattr(self, "_augumenter_list"):
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def fit(self, raw_signals: RawSignals, y=None):
        """
        Intentionally does nothing
        """
        self._prepare_effective_augumenter_list()

        for aug in self._augumenter_list:
            aug.fit(raw_signals, y)
        return self

    def _inner_transform(self, raw_signals: RawSignals) -> RawSignals:
        """Apply all augmenters and concatenate their outputs.

        Args:
            raw_signals: The input dataset.

        Returns:
            Combined augmented :class:`RawSignals`.
        """
        new_signals = raw_signals.initialize_empty()

        for aug in self._augumenter_list:
            new_signals += aug.transform(raw_signals)

        return new_signals

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        """Transform the dataset by applying all augmenters in parallel.

        Args:
            raw_signals: The dataset to augment.

        Returns:
            Augmented :class:`RawSignals`, optionally including originals.
        """
        self._check_fitted()

        new_signals = self._inner_transform(raw_signals)

        if self.append_original:
            new_signals += raw_signals

        return new_signals

    def fit_transform(self, raw_signals: RawSignals, y=None) -> RawSignals:
        """Fit all augmenters and transform the dataset.

        Args:
            raw_signals: The dataset to augment.
            y: Ignored. Present for API compatibility.

        Returns:
            Augmented :class:`RawSignals`.
        """
        self.fit(raw_signals, y)
        return self.transform(raw_signals)

    def sample(self, raw_signals: RawSignals, n_samples: int = 1) -> RawSignals:
        """Augment and then randomly sample from the combined output.

        Args:
            raw_signals: The dataset to augment and sample from.
            n_samples: Number of samples to draw.

        Returns:
            A new :class:`RawSignals` with ``n_samples`` augmented signals.
        """
        self._check_fitted()
        
        new_sigsnals_pre = self._inner_transform(raw_signals)
        n_new_signals = len(new_sigsnals_pre)
        replace = n_samples > n_new_signals
        indices = np.random.choice(n_new_signals, size=n_samples, replace=replace)
        new_signals = new_sigsnals_pre.initialize_empty()
        for idx in indices:
            new_signals += [new_sigsnals_pre[idx]]
            
        return new_signals