"""Module defining the abstract interface for raw signal creators.

Provides :class:`RawSignalsCreator`, the base contract for all
synthetic signal dataset generators.
"""
import abc
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals

class RawSignalsCreator(abc.ABC):

    """Abstract interface for synthetic raw signal dataset generators."""

    @abc.abstractmethod
    def get_set(self) ->RawSignals:
        """
        Returns:
        --------
        RawSignals dataset
        """
