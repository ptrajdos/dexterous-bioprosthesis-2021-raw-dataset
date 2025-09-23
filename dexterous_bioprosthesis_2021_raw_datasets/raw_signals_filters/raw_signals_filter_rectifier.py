import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

from copy import deepcopy


class RawSignalsFilterRectifier(RawSignalsFilter):

    def fit(self, raw_signals: RawSignals):
        return self

    def transform(self, raw_signals: RawSignals):
        
        copied_signals = deepcopy(raw_signals)
        for r_signal in copied_signals:
            r_signal.signal = np.abs(r_signal.signal)

        return copied_signals
