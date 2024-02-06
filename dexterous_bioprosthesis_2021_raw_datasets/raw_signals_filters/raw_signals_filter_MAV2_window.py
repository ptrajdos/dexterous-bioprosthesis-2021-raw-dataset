

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window import RawSignalsFilterWindowFilter

class RawSignalsFilterMAV2WindowFilter(RawSignalsFilterWindowFilter):
    
    def __init__(self, window_length: int = 100) -> None:
        super().__init__(window_length)

    def channel_transform(self,data):
        N = self.window_length
        M = N//2
        out = np.zeros(data.shape[0])
        for i in range(M, out.shape[0]-M):
            out[i] = np.sum( np.abs(data[i-M:i+M]))/N
        return out