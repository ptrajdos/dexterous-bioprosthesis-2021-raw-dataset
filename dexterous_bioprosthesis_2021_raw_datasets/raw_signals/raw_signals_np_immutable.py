from copy import deepcopy
import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.iraw_signals import (
    IRawSignals,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from .raw_signal import RawSignal
from collections.abc import Iterable
from collections.abc import Collection


class RawSignalsNpImmutable(IRawSignals):
    """
    Class represents a dataset of raw signals.
    It assumes that the underlying numpy array is immutable. You cannot add or remove signals after creation.
    All signals must have the same number of channels and length.
    All signals mus thave the same sample rate.
    """

    @classmethod
    def from_numpy(
        cls,
        raw_signals_buffer,
        raw_signals_labels,
        raw_signals_timestamps,
        sample_rate=1000,
    ):
        """
        Creates RawSignalsNpImmutable object from numpy arrays.

        Arguments:
        ----------
        raw_signals_buffer -- numpy array of shape (n_objects, n_samples, n_channels)
        raw_signals_labels -- numpy array of shape (n_objects,)
        raw_signals_timestamps -- numpy array of shape (n_objects,)
        sample_rate -- sample rate of the signals

        Returns:
        --------
        RawSignalsNpImmutable object
        """
        obj = cls()
        obj.raw_signals_buffer = raw_signals_buffer
        obj.raw_signals_labels = raw_signals_labels
        obj.raw_signals_timestamps = raw_signals_timestamps
        obj.sample_rate = sample_rate
        return obj

    def __init__(self, raw_signal_list=None, sample_rate=1000) -> None:
        """
        Creates a new instance of the class
        """
        tmp_raw_signals = RawSignals(
            raw_signal_list=raw_signal_list, sample_rate=sample_rate
        )

        self.raw_signals_buffer = tmp_raw_signals.to_numpy()
        self.raw_signals_labels = tmp_raw_signals.get_labels()
        self.raw_signals_timestamps = tmp_raw_signals.get_timestamps()
        # TODO what about channel names?

        self.sample_rate = sample_rate

    def __iter__(self):
        for signal, label, timestamp in zip(
            iter(self.raw_signals_buffer),
            iter(self.raw_signals_labels),
            iter(self.raw_signals_timestamps),
        ):
            yield RawSignal(
                signal=signal,
                object_class=label,
                timestamp=timestamp,
                sample_rate=self.sample_rate,
            )

    def __getitem__(self, key):

        if not isinstance(key, tuple):
            if (
                isinstance(key, slice)
                or isinstance(key, Collection)
                or isinstance(key, Iterable)
            ):

                if isinstance(key, Iterable):
                    key = list(key)

                return RawSignalsNpImmutable.from_numpy(
                    raw_signals_buffer=self.raw_signals_buffer[key],
                    raw_signals_labels=self.raw_signals_labels[key],
                    raw_signals_timestamps=self.raw_signals_timestamps[key],
                    sample_rate=self.sample_rate,
                )

            # single item!
            return RawSignal(
                signal=self.raw_signals_buffer[key],
                object_class=self.raw_signals_labels[key],
                timestamp=self.raw_signals_timestamps[key],
                sample_rate=self.sample_rate,
            )

        # Is tuple here

        if len(key) == 1:
            return self.__getitem__(key[0])

        selected_signals = self.__getitem__(key[0])

        if isinstance(selected_signals, RawSignal):
            return selected_signals[key[1:]]

        r_key = key[1:]
        return RawSignalsNpImmutable.from_numpy(
            raw_signals_buffer=(
                selected_signals.raw_signals_buffer[:, r_key[0]]
                if len(r_key) == 1
                else selected_signals.raw_signals_buffer[:, r_key[0], r_key[1]                    
                ]
            ),
            raw_signals_labels=selected_signals.raw_signals_labels,
            raw_signals_timestamps=selected_signals.raw_signals_timestamps,
            sample_rate=self.sample_rate,
        )

    def append(self, other: RawSignal):

        raise NotImplementedError(
            "RawSignalsNpImmutable object is immutable. You cannot append new signals."
        )

    def __iadd__(self, other):
        raise NotImplementedError(
            "RawSignalsNpImmutable object is immutable. You cannot append new signals."
        )

    def __len__(self):
        return self.raw_signals_buffer.shape[0]

    def __eq__(self, __o: object) -> bool:

        if id(self) == id(__o):
            return True

        if type(self) != type(__o):
            return False

        if self.sample_rate != __o.sample_rate:
            return False
    
        if len(self) != len(__o):
            return False
        
        for i1, i2 in zip(iter(self), iter(__o)):
            if i1 != i2:
                return False
            
        return True

    def get_labels(self):
        """
        Returns:
        -------

        List of labels of stored signals
        """

        return self.raw_signals_labels

    def get_timestamps(self):
        """
        Returns:
        --------

        List of timestamps of the stored signals.

        """
        return self.raw_signals_timestamps

    def set_labels(self, labels):
        raise NotImplementedError(
            "RawSignalsNpImmutable object is immutable. You cannot set new labels."
        )

    
    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, sample_rate):
        raise NotImplementedError(
            "RawSignalsNpImmutable object is immutable. You cannot set new sample rate."
        )

    def to_numpy(self):
        """
        Returns the raw_signals object as a numpy array.
        Assuming that all raw_signal object have the same dimension

        Returns:
        ---------
        numpy array of shape (n_objects, n_samples, n_channels)

        """

        return self.raw_signals_buffer

    def to_numpy_concat(self):
        """
        Returns the raw_signals object as a concatenated numpy array.
        Assuming that all raw_signal object have the same number of channels

        Returns:
        ---------
        numpy array of shape (sum(n_samples), n_channels)

        """
        np_array =  np.reshape(self.raw_signals_buffer, (-1, self.raw_signals_buffer.shape[2]))
        return np_array
