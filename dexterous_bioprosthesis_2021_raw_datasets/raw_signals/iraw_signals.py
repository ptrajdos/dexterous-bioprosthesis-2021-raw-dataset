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
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_labels(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_timestamps(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_labels(self, labels):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_sample_rate(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_sample_rate(self, sample_rate):
        raise NotImplementedError
    
    @abc.abstractmethod
    def to_numpy(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def to_numpy_concat(self):
        raise NotImplementedError