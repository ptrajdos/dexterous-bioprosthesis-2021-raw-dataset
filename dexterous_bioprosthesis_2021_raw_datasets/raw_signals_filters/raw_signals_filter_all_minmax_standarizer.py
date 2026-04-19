from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterAllMinmaxStandarizer(RawSignalsFilter):

    def __init__(self, range_min=0, range_max=1, eps=1e-30) -> None:
        super().__init__()
        self.range_min = range_min
        self.range_max = range_max
        self.eps = eps

    def fit(self, raw_signals: RawSignals):
        self._min = raw_signals[0].to_numpy().min()
        self._max = raw_signals[0].to_numpy().max()

        for rs in raw_signals:
            t_min = rs.to_numpy().min()
            t_max = rs.to_numpy().max()
            self._min = np.min([self._min, t_min])
            self._max = np.max([self._max, t_max])

        return super().fit(raw_signals)

    def transform(self, raw_signals: RawSignals):
        self._check_fitted()
        copied_signals = deepcopy(raw_signals)
        for r_signal in copied_signals:
            r_signal.signal = (
                (r_signal.signal - self._min) / (self._max - self._min + self.eps)
            ) * (self.range_max - self.range_min) + self.range_min

        return copied_signals
