
import abc
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals

class DistanceMatrixCalculator(abc.ABC):
    """
    An interace for objects that calculates pairwise distance matrix for raw signals.
    """

    @abc.abstractmethod
    def calculate_distance_matrix(self, raw_signals:RawSignals):
        """
        Calculates the pairwise distance matrix for raw signals.

        Arguments:
        ---------

        raw_signals -- raw signals to calculate the distance matrix

        Returns:
        -------
        Distance matrix as numpy 3D array (n_channels, n_objects, n_objects)
        """

    @abc.abstractmethod
    def raw_signal_dist(self, raw_signal_a: RawSignal, raw_signal_b:RawSignal):
        """
        Calculates a distance between raw signals.

        Arguments:
        ----------
        raw_signal_a: RawSignal -- first raw signal object.
        raw_signal_b: RawSignal -- second raw signal object.

        Returns:
        -------
        Distance vector between the raw signals (n_channels)
        """

    @abc.abstractmethod
    def raw_signal_dist_2_set(self, raw_signal: RawSignal, raw_signals:RawSignals):
        """
        Calculates distances between raw_signal and a set of raw_signals.

        Arguments:
        ----------
        raw_signal: RawSignal -- raw signal object.
        raw_signals: RawSignals -- raw signals object.

        Returns:
        -------
        Distance vector between the raw signals (n_channels, n_objects)
        """

    @abc.abstractmethod
    def calculate_distance_matrix_set_2_set(self, raw_signals_1:RawSignals, raw_signals_2:RawSignals):
        """
        Calculates the pairwise distance matrix for two raw signals.

        Arguments:
        ---------

        raw_signals_1 -- first set of raw signals to calculate the distance matrix
        raw_signals_2 -- second set of raw signals to calculate the distance matrix

        Returns:
        -------
        Distance matrix as numpy 3D array (n_channels, n_objects_1, n_objects_2)
        """