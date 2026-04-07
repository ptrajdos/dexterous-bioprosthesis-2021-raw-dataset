from __future__ import annotations
import abc

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsSpoilerInterface(abc.ABC):

    @abc.abstractmethod
    def fit(self, raw_signals: RawSignals) -> RawSignalsSpoilerInterface:
        """
        Fits the RawSignalsSpoiler

        Arguments:
        ----------
        raw_signals -- raw_signals to fit

        Returns:
          RawSignalSpoiler (self)
        """

    @abc.abstractmethod
    def transform(self, raw_signals: RawSignals) -> RawSignals:
        """
        Transforms the RawSignals

        Arguments:
        ----------
        raw_signals:RawSignals -- raw_signals to transform

        Returns:
        --------
            RawSignals -- transformed signals

        """

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        """
        Fits and then transforms the RawSignals

        Arguments:
        ----------
        raw_signals:RawSignals -- raw_signals to transform

        Returns:
        --------
            RawSignals -- transformed signals

        """
        return self.fit(raw_signals).transform(raw_signals)
