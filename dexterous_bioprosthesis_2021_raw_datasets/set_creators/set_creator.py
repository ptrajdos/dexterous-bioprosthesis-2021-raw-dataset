"""Module defining the abstract interface for set creators.

Provides :class:`SetCreator`, the base contract for transforming
raw signal collections into feature matrices, and :class:`ASetCreator`,
a convenience base class that stores channel names during fit.
"""
from __future__ import annotations
import abc
from typing import Any, Optional, Tuple
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class SetCreator(abc.ABC):
    """An interface for creating ordinary datasets from raw signal datasets
    """

    @abc.abstractmethod
    def fit_transform(self, raw_signals: RawSignals, y=None) -> tuple:
        """Create dataset from raw signals.
        Fits model if necesary and then transforms the raw_signals.

        Arguments:
        ---------
        raw_signals -- A RawSignals object
        y -- ignored

        Returns:
        -------
        A touple containig:
        X -- data
        y -- classes
        t -- timestamp

        """

    @abc.abstractmethod
    def fit(self, raw_signals: RawSignals, y=None) -> SetCreator:
        """Only fit the transformation model.

        Arguments:
        ---------
        raw_signals -- A RawSignals object
        y -- ignored

        Returns:
        SetCreator

        """

    @abc.abstractmethod
    def transform(self, raw_signals: RawSignals) -> tuple:
        """Create dataset from raw signals.
        Transform the raw_signals using previously fitted model.

        Arguments:
        ---------
        raw_signals -- A RawSignals object

        Returns:
        -------
        A touple containig:
        X -- data
        y -- classes
        t -- timestamp

        """

    @abc.abstractmethod
    def get_channel_attribs_indices(self) -> Optional[list[Any]]:
        """Get indices of channel-specific attributes

        Returns:
        List containing lists of channel specific attributes (their indices).
        Or None.
        None means that there is no simple mapping from channels to attributes in output set

        """

    @abc.abstractmethod
    def get_channel_names(self)->Tuple[str]:
        """
        Get channel names.

        Returns:
        --------
        channel names as a list/tuple
        """


class ASetCreator(SetCreator):
    """Abstract base class that implements common fit logic for set creators.

    Stores channel names from the raw signals during :meth:`fit` and
    provides :meth:`get_channel_names`.
    Subclasses must call ``super().fit(raw_signals)`` in their own ``fit``
    implementation.
    """

    def __init__(self) -> None:
        self._channel_names: Tuple[str] = tuple()

    def fit(self, raw_signals: RawSignals, y=None) -> ASetCreator:
        """Store channel names from the raw signals.

        Subclasses should call ``super().fit(raw_signals)`` to ensure
        channel names are recorded.

        Arguments:
        ---------
        raw_signals -- A RawSignals object
        y -- ignored

        Returns:
        --------
        self
        """
        raw_signals = RawSignals.construct_from_list(raw_signals)
        self._channel_names = raw_signals.get_channel_names()
        return self

    def get_channel_names(self) -> Tuple[str]:
        """
        Get channel names.

        Returns:
        --------
        channel names as a list/tuple
        """
        return self._channel_names
