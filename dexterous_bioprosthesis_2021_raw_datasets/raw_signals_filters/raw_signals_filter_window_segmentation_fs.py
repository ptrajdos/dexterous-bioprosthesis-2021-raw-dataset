from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

from copy import deepcopy


class RawSignalsFilterWindowSegmentationFS(RawSignalsFilter):

    def __init__(self, window_length:float, overlap:float ) -> None:
        """
        Segment the signal using sliding window.
        New signals appear in the object

        Arguments:
        ----------
        window_length:float --  window length in ms
        overlap:float -- overlap in ms
         
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
        fs = raw_signals.sample_rate

        window_length_samples = int(self.window_length * fs / 1000)
        overlap_samples = int(self.overlap * fs / 1000)


        for signal in raw_signals:
            s_len = signal.signal.shape[0]
            start_idx = 0
            end_idx = start_idx + window_length_samples
            while end_idx <= s_len:
                copied_signal = signal[start_idx:end_idx, :]
                copied_signal.signal = copied_signal.signal.copy()
                new_signals.append(copied_signal)

                start_idx += window_length_samples - overlap_samples
                end_idx += window_length_samples - overlap_samples

        return new_signals