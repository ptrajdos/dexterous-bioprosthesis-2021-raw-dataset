import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

from copy import deepcopy


class RawSignalsFilterAllRobustStandarizer(RawSignalsFilter):

   

    def fit(self, raw_signals: RawSignals):
        np_data_all = raw_signals.to_numpy_concat()
        self._median = np.median(np_data_all)
        q1, q3 = np.percentile(np_data_all, [25, 75])
        self._iqr = q3 - q1
        self._fitted = True
        return self

    def transform(self, raw_signals: RawSignals):
        if not hasattr(self, '_fitted'):
            raise RuntimeError("Filter not fitted. Call 'fit' with training data before using this method.")

        copied_signals = deepcopy(raw_signals)
        for r_signal in copied_signals:
            r_signal.signal = (r_signal.signal - self._median)/self._iqr

        return copied_signals
