from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterLengthMedianStandarizer(RawSignalsFilter):
    """
    Forces all filtered signals to have length equal to median of signal lengths in the training data
    """

    def fit(self, raw_signals: RawSignals):
        super().fit(raw_signals)
        lengths = [rs.to_numpy().shape[0] for rs in raw_signals]
        self._median = int( np.median(lengths))

        return self

    def transform(self, raw_signals: RawSignals):
        self._check_fitted()

        new_signals = raw_signals.initialize_empty()
        for r_signal in raw_signals:
            r_sig_len = len(r_signal)
            r_sig_np = r_signal.to_numpy()

            if r_sig_len > self._median:
                r_signal.signal = r_sig_np[:self._median,:]

            else:
                pad =  int(self._median - r_sig_len)
                r_signal.signal = np.pad(r_sig_np, ((0,pad),(0,0)), mode='constant')
            
            new_signals.append(r_signal)

        return new_signals
