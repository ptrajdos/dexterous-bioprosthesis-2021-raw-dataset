from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterAllRobustStandarizer(RawSignalsFilter):

    def __init__(self, eps=1e-30) -> None:
        super().__init__()
        self.eps = eps

    def fit(self, raw_signals: RawSignals):
        np_data_all = raw_signals.to_numpy_concat()
        dtype = np_data_all.dtype
        self._median = np.median(np_data_all).astype(dtype)
        q1, q3 = np.percentile(np_data_all, [25, 75]).astype(dtype)
        self._iqr = q3 - q1
        if self._iqr < self.eps:
            self._iqr = self.eps
        
        return super().fit(raw_signals)

    def transform(self, raw_signals: RawSignals):
        self._check_fitted()
        
        copied_signals = deepcopy(raw_signals)
        for r_signal in copied_signals:
            r_signal.signal = (r_signal.signal - self._median) / self._iqr

        return copied_signals
