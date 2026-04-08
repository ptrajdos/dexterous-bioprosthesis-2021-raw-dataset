from __future__ import annotations
import abc

from sklearn.exceptions import NotFittedError

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsFilter(abc.ABC):
    """
    Class that represents an interface for filters
    """

    @abc.abstractmethod
    def fit(self, raw_signals: RawSignals) -> RawSignalsFilter:
        """
        Fits the filter.
        Arguments:
        ---------
        raw_signals --- An object of RawSignals class to be fitted with
        """
        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this filter."
            )

    @abc.abstractmethod
    def transform(self, raw_signals: RawSignals) -> RawSignals:
        """
        Transforms the RawSignals object
        Arguments:
        ---------
        raw_signals --- RawSignals object to be filtered

        Returns:
        --------
        RawSignals object
        """

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        """
        Fits and then transforms the given RawSignals object

        Arguments:
        ---------
        raw_signals --- RawSignals object to be filtered

        Returns:
        --------
        RawSignals object
        """
        self.fit(raw_signals)
        return self.transform(raw_signals)
