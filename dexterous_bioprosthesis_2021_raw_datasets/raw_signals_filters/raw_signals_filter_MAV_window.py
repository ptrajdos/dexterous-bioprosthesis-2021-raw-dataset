

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window import RawSignalsFilterWindowFilter

class RawSignalsFilterMAVWindowFilter(RawSignalsFilterWindowFilter):
    
    def __init__(self, window_length: int = 100) -> None:
        super().__init__(window_length)

    def channel_transform(self,data):
        M = self.window_length//2

        out = np.zeros(data.shape[0])
        mask = np.ones(self.window_length)

        for i in range(len(mask)):
            if i<0.25*self.window_length:
                mask[i] = 4*i/self.window_length
            if i>0.75*self.window_length:
                mask[i] = 4*(self.window_length-i)/self.window_length

        for i in range(M, out.shape[0]-M):
            out[i] = np.sum( np.multiply(np.abs(data[i-M:i+M]),mask ))/self.window_length

        return out