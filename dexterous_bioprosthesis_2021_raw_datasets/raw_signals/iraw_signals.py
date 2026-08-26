"""Module defining the abstract interface for raw signal collections.

Provides :class:`IRawSignals`, the base contract that all raw signal
collection implementations must satisfy.
"""
import abc

class IRawSignals(abc.ABC):
    """
    Interface for raw signals dataset.
    """

    def __iter__(self):
        raise NotImplementedError

    def __iadd__(self, other):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def __eq__(self, __o: object) -> bool:
        raise NotImplementedError
    
    def __getitem__(self,key):
        raise NotImplementedError

    @abc.abstractmethod
    def append(self, other):
        """Append a signal to the collection."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_labels(self):
        """Return the labels of all stored signals."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_timestamps(self):
        """Return the timestamps of all stored signals."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_labels(self, labels):
        """Set new labels for all stored signals."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_sample_rate(self):
        """Return the sample rate."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_sample_rate(self, sample_rate):
        """Set the sample rate."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def to_numpy(self):
        """Return the signal data as a numpy array."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def to_numpy_concat(self):
        """Return all signals concatenated as a single numpy array."""
        raise NotImplementedError