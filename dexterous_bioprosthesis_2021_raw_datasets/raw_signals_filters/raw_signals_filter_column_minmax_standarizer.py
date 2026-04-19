from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterColumnMinMaxStandarizer(RawSignalsFilter):

    def __init__(self, eps=1e-30, range_min=0, range_max=1) -> None:
        super().__init__()
        self.eps = eps
        self.range_min = range_min
        self.range_max = range_max

    def fit(self, raw_signals: RawSignals):

        np_data = raw_signals.to_numpy_concat()
        self._min = np.min(np_data, axis=(0, 1))
        self._max = np.max(np_data, axis=(0, 1))

        return super().fit(raw_signals)

    def transform(self, raw_signals: RawSignals):

        self._check_fitted()

        copied_signals = deepcopy(raw_signals)
        for r_signal in copied_signals:
            sig_dtype = r_signal.to_numpy().dtype
            r_signal.signal = (
                ((r_signal.signal - self._min) / (self._max - self._min + self.eps))
                * (self.range_max - self.range_min)
                + self.range_min
            ).astype(sig_dtype)

        return copied_signals
