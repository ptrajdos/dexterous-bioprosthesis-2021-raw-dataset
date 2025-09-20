from copy import deepcopy

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

class RawSignalsFilterFreqThreshold(RawSignalsFilter):

    def __init__(self, threshold=0.1) -> None:
        super().__init__()
        self.threshold = threshold
        


    def fit(self, raw_signals: RawSignals) -> None:
        return super().fit(raw_signals)
    
    

    def transform(self,raw_signals: RawSignals)->RawSignals:
        

        copied_signals = deepcopy(raw_signals)

        for raw_signal in copied_signals:
            sig_np = raw_signal.signal
            n_samples, n_ch = sig_np.shape

            for ch_idx in range(n_ch):
                fft_vals = np.fft.rfft(sig_np[:,ch_idx])
                magnitude = np.abs(fft_vals)

                # Thresholding
                threshold = self.threshold * np.max(magnitude)
                fft_filtered = fft_vals * (magnitude > threshold)
                filtered_signal = np.fft.irfft(fft_filtered, n=n_samples)
                sig_np[:,ch_idx] = filtered_signal

        return copied_signals