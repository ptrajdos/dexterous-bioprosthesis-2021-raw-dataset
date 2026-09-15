import abc
from typing import List, Tuple, Union

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals

class IRawSignalsSplitter(abc.ABC):

    @abc.abstractmethod
    def split_signals(self, raw_signals:RawSignals)->Union[List[RawSignals], Tuple[RawSignals]]:
        """
        Splint one raw_signals object into multiple sub_sets.

        Arguments:
        ----------

        raw_signals -- input set

        Returns:
        --------
        list/tuple of RawSignals
        """