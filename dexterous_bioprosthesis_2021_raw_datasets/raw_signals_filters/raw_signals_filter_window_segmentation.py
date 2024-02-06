from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

from copy import deepcopy
from scipy.signal import decimate

class RawSignalsFilterWindowSegmentation(RawSignalsFilter):

    def __init__(self, window_length:int, overlap:int ) -> None:
        """
        Segment the signal using sliding window.
        New signals appear in the object

        Arguments:
        ----------
        start_sample:int statr index of the window (included)
        end_sample:int end index of the window (not included)
         
        """
        super().__init__()
        self.window_length = window_length
        self.overlap = overlap
    
    def fit(self, raw_signals:RawSignals):
        """
        Does nothing
        """
        pass

    def transform(self,raw_signals:RawSignals):
        """
        Apply windowed segmentation with overlap
        """
        new_signals = RawSignals(sample_rate=raw_signals.sample_rate)

        for signal in raw_signals:
            s_len = signal.signal.shape[0]
            start_idx = 0
            end_idx = start_idx + self.window_length
            while end_idx <= s_len:
                copied_signal = deepcopy(signal)
                copied_signal.signal = copied_signal.signal[ start_idx:end_idx, :]
                new_signals.append(copied_signal)

                start_idx += self.window_length - self.overlap
                end_idx += self.window_length - self.overlap

        return new_signals