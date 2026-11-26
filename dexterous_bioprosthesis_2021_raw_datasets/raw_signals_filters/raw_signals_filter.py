"""Module defining the abstract interface for raw signal filters.

Provides :class:`RawSignalsFilter`, the base contract with ``fit``,
``transform``, and ``fit_transform`` methods following the scikit-learn
transformer pattern.
"""
from __future__ import annotations
import abc

import numpy as np

from sklearn.base import BaseEstimator

from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.tools.sklearn_compat import sklearn_transformer_tags

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsFilter(BaseEstimator, abc.ABC):
    """Class that represents an interface for filters
    """

    def __sklearn_tags__(self):
        """scikit-learn >= 1.6 tags."""
        return sklearn_transformer_tags()

    def __sklearn_is_fitted__(self) -> bool:
        """Return True if fitted. Used by sklearn.utils.validation.check_is_fitted."""
        return getattr(self, "_is_fitted", False)

    @abc.abstractmethod
    def fit(self, raw_signals: RawSignals, y=None) -> RawSignalsFilter:
        """Fits the filter.

        Arguments:
        ---------
        raw_signals --- An object of RawSignals class to be fitted with
        y --- not used, only for compatibility with sklearn pipeline

        """
        self._is_fitted = True
        return self

    @staticmethod
    def _shifted_timestamp(raw_signal: RawSignal, row_key, sample_rate=None):
        """Timestamp of the first sample selected by ``row_key``.

        Arguments:
        ---------
        raw_signal -- the original (uncut) RawSignal
        row_key -- int, slice, list of indices or boolean mask selecting rows
        sample_rate -- sample rate to use; defaults to the signal's sample rate

        Returns:
        -------
        The original timestamp (seconds) shifted by the offset of the first
        selected row. If nothing is selected, the original timestamp is returned.

        """
        fs = raw_signal.sample_rate if sample_rate is None else sample_rate
        selected = np.atleast_1d(np.arange(raw_signal.signal.shape[0])[row_key])
        if selected.size == 0:
            return raw_signal.timestamp
        return raw_signal.timestamp + int(selected[0]) / fs

    def _check_fitted(self):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this filter."
            )

    @abc.abstractmethod
    def transform(self, raw_signals: RawSignals) -> RawSignals:
        """Transforms the RawSignals object

        Arguments:
        ---------
        raw_signals --- RawSignals object to be filtered

        Returns:
        -------
        RawSignals object

        """

    def fit_transform(self, raw_signals: RawSignals, y=None) -> RawSignals:
        """Fits and then transforms the given RawSignals object

        Arguments:
        ---------
        raw_signals --- RawSignals object to be filtered
        y --- not used, only for compatibility with sklearn pipeline

        Returns:
        -------
        RawSignals object

        """
        self.fit(raw_signals, y)
        return self.transform(raw_signals)
