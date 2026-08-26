"""Module providing the base implementation for single-signal augmenters.

This module contains :class:`RawSignalsAugumenterBase`, which extends
:class:`RawSignalsAugumenter` with parallel processing support, random state
management, and a template-method pattern where subclasses only need to
implement :meth:`_sig_augument` for a single signal.
"""
import abc

from joblib import delayed
import numpy as np
from sklearn.utils import check_random_state
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.tools.progressparallel import (
    ProgressParallel,
)


class RawSignalsAugumenterBase(RawSignalsAugumenter):
    """Base class for augmenters that operate on individual signals.

    Provides parallel execution via joblib, optional appending of original
    signals, configurable repetition count, and reproducible random state.
    Subclasses must implement :meth:`_sig_augument` to define the
    augmentation logic for a single :class:`RawSignal`.

    Args:
        n_jobs: Number of parallel jobs for joblib. ``None`` means sequential.
        append_original: If ``True``, original signals are appended to the
            augmented output.
        n_repeats: Number of augmented copies to create per signal.
        random_state: Seed or :class:`numpy.random.RandomState` instance for
            reproducibility.
    """

    def __init__(self, n_jobs=None, append_original=True, n_repeats:int=1, random_state=10) -> None:
        super().__init__()
        self.n_jobs = n_jobs
        self.append_original:bool = append_original
        self.n_repeats:int = n_repeats
        self.random_state = random_state

    @abc.abstractmethod
    def _sig_augument(self, raw_signal: RawSignal, n_repeats: int=1) -> list:
        """
        Auguments a single signal

        Arguments:
        ---------
        raw_signal: RawSignal -- the signal to be augumented
        n_repeats: int -- how many augumented versions of signal to create

        Returns:
        --------
        List of augumented signals

        """
    
    def _check_if_fitted(self):
        """Raise :class:`NotFittedError` if the augmenter has not been fitted."""
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        """Augment all signals in the dataset using parallel processing.

        Args:
            raw_signals: The dataset of raw signals to augment.

        Returns:
            A new :class:`RawSignals` containing augmented (and optionally
            original) signals.
        """
        self._check_if_fitted()
        new_signals = raw_signals.initialize_empty()

        aug_sig_list = ProgressParallel(
            n_jobs=self.n_jobs, use_tqdm=True, total=len(raw_signals)
        )(delayed(self._sig_augument)(sig, self.n_repeats) for sig in raw_signals)

        for aug_sigs in aug_sig_list:
            new_signals += aug_sigs

        if self.append_original:
            new_signals += raw_signals

        return new_signals

    def fit_transform(self, raw_signals: RawSignals, y=None) -> RawSignals:
        """Fit the augmenter and then transform the dataset.

        Args:
            raw_signals: The dataset of raw signals.
            y: Ignored. Present for scikit-learn pipeline compatibility.

        Returns:
            Augmented :class:`RawSignals`.
        """
        self.fit(raw_signals, y)
        return self.transform(raw_signals)

    def fit(self, raw_signals: RawSignals, y=None) -> RawSignalsAugumenter:
        """Fit the augmenter by initialising the random state.

        Args:
            raw_signals: The dataset of raw signals.
            y: Ignored. Present for scikit-learn pipeline compatibility.

        Returns:
            The fitted augmenter instance.
        """
        self._is_fitted = True
        self._random_state = check_random_state(self.random_state)
        return self
    
    def sample(self, raw_signals: RawSignals, n_samples: int=1) -> RawSignals:
        """Randomly sample and augment signals from the dataset.

        Args:
            raw_signals: The dataset to sample from.
            n_samples: Number of augmented samples to produce.

        Returns:
            A new :class:`RawSignals` with ``n_samples`` augmented signals.
        """
        self._check_if_fitted()
        n_signals = len(raw_signals)
        replace = n_samples > n_signals
        indices = self._random_state.choice(len(raw_signals), size=n_samples, replace=replace)
        new_signals = raw_signals.initialize_empty()
        sel_signals = raw_signals[indices]

        for sig in sel_signals:
            new_signals += self._sig_augument(sig, n_repeats=1)
            
        return new_signals
