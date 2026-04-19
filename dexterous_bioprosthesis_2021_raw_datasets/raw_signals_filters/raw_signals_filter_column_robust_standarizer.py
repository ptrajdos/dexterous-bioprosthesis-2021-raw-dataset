from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterColumnRobustStandarizer(RawSignalsFilter):

    def __init__(self, eps=1e-30) -> None:
        super().__init__()
        self.eps = eps

    def fit(self, raw_signals: RawSignals):
        np_data = raw_signals.to_numpy_concat()
        q1, self._median, q3 = np.percentile(np_data, [25, 50, 75], axis=(0, 1))
        self._iqr = q3 - q1

        return super().fit(raw_signals)

    def transform(self, raw_signals: RawSignals):

        self._check_fitted()

        copied_signals = deepcopy(raw_signals)
        for r_signal in copied_signals:
            sig_dtype = r_signal.to_numpy().dtype
            r_signal.signal = ((r_signal.signal - self._median) / (self._iqr + self.eps)).astype(sig_dtype)

        return copied_signals
