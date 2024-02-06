from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

from copy import deepcopy
from scipy.signal import decimate

class RawSignalsFilterWindowSingle(RawSignalsFilter):

    def __init__(self, start_sample:int, end_sample:int ) -> None:
        """
        Cuts single window from given RawSignals.

        Arguments:
        ----------
        start_sample:int statr index of the window (included)
        end_sample:int end index of the window (not included)
         
        """
        super().__init__()
        self.start_sample = start_sample
        self.end_sample = end_sample
    
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

        for signal in copied_signals:
            signal.signal = signal.signal[self.start_sample:self.end_sample, :]


        return copied_signals