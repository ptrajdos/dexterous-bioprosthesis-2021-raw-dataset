from copy import deepcopy
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

from scipy import signal
import scipy


class RawSignalsFilterButter(RawSignalsFilter):

    def __init__(self, low_freq=48, high_freq=52, order=4, btype='bandstop') -> None:
        super().__init__()
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        self.btype = btype


    def fit(self, raw_signals: RawSignals) -> None:
        return super().fit(raw_signals)
    

    def transform(self,raw_signals: RawSignals)->RawSignals:
        nyq = raw_signals.sample_rate * 0.5
        low = self.low_freq/ nyq
        high = self.high_freq/nyq

        copied_signals = deepcopy(raw_signals)

        for raw_signal in copied_signals:
            sig_np = raw_signal.signal
            n_ch = sig_np.shape[1]

            for ch_idx in range(n_ch):
                b, a = signal.butter(self.order, [low, high], btype=self.btype)
                sig_np[:,ch_idx] = signal.filtfilt(b,a,sig_np[:,ch_idx])

        return copied_signals