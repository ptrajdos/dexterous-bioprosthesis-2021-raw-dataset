from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

from copy import deepcopy
from scipy.signal import decimate

class RawSignalsFilterDecimate(RawSignalsFilter):

    def __init__(self, downsample_factor=2) -> None:
        super().__init__()
        self.downsample_factor = downsample_factor
    
    def fit(self, raw_signals:RawSignals):
        """
        Does nothing
        """
        pass

    def transform(self,raw_signals:RawSignals):
        """
        Decimates signals
        """
        copied_signals = deepcopy(raw_signals)

        copied_signals.sample_rate = copied_signals.sample_rate//self.downsample_factor

        for signal in copied_signals:
            signal.signal = decimate(signal.signal, q= self.downsample_factor, axis=0).astype(signal.signal.dtype)


        return copied_signals