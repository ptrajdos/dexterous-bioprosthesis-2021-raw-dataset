"""Module defining the abstract base interface for raw signal augmenters.

This module provides the :class:`RawSignalsAugumenter` abstract class that
establishes the contract all concrete augmenters must follow. The interface
is designed to be compatible with scikit-learn pipelines.
"""
from __future__ import annotations
import abc

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsAugumenter(abc.ABC):
    """Abstract base class defining the interface for raw signal augmenters.

    All concrete augmenters must implement :meth:`fit`, :meth:`transform`,
    :meth:`fit_transform`, and :meth:`sample` methods. This ensures a
    consistent API across all augmentation strategies.
    """

    @abc.abstractmethod
    def fit(self, raw_signals: RawSignals, y=None) -> RawSignalsAugumenter:
        """
        Fits the data augumenter

        Arguments:
        ---------
        raw_signals: RawSignals -- the dataset to be augumented
        y: not used, only for compatibility with sklearn pipeline

        """

    @abc.abstractmethod
    def transform(self, raw_signals: RawSignals) -> RawSignals:
        """
        Transforms the dataset.
        New data contains changed version of each RawSignal in RawSignals

        Arguments:
        ---------
        raw_signals: RawSignals -- the dataset to be augumented

        Returns:
        --------
        Transformed RawSignals

        """

    @abc.abstractmethod
    def fit_transform(self, raw_signals: RawSignals, y=None) -> RawSignals:
        """
        Fits and then transforms the dataset.
        New data contains changed version of each RawSignal in RawSignals

        Arguments:
        ---------
        raw_signals: RawSignals -- the dataset to be augumented
        y: not used, only for compatibility with sklearn pipeline

        Returns:
        --------
        Transformed RawSignals

        """

    @abc.abstractmethod
    def sample(self, raw_signals: RawSignals, n_samples: int=1) -> RawSignals:
        """
        Samples n_samples from the dataset

        Arguments:
        ---------
        raw_signals: RawSignals -- the dataset to be sampled
        n_samples: int -- how many samples to sample

        Returns:
        --------
        Sampled RawSignals

        """
