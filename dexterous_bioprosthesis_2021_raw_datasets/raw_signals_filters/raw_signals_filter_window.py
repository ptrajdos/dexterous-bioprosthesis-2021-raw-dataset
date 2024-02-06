

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

import numpy as np
from copy import deepcopy
import abc

class RawSignalsFilterWindowFilter(RawSignalsFilter):
    
    def __init__(self, window_length:int=100) -> None:
        super().__init__()
        self.window_length = window_length


    def fit(self, raw_signals: RawSignals) -> None:
        """
        Does nothing
        """
        return super().fit(raw_signals)
    @abc.abstractmethod
    def channel_transform(self,data):
        """
        Transforms single channel

        Arguments:
        data -- one dimensional numpy array

        Returns:
        one domensional numpy array -- filtered channel
        """

    
    def transform(self,raw_signals: RawSignals)->RawSignals:
        
        copied_signals = deepcopy(raw_signals)

        for signal in copied_signals:

            n_ch = signal.signal.shape[1]

            for ch_idx in range(n_ch):
                signal.signal[:,ch_idx] = self.channel_transform(signal.signal[:,ch_idx])

        return copied_signals
        

