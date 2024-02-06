import abc
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals

class RawSignalsCreator(abc.ABC):

    @abc.abstractmethod
    def get_set(self) ->RawSignals:
        """
        Returns:
        --------
        RawSignals dataset
        """
