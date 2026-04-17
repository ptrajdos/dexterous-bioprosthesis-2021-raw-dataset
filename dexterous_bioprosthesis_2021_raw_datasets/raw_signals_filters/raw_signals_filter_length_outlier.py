from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterLengthOutlier(RawSignalsFilter):
    """
    Filters out raw signals objects with outlying lengths (number of samples)
    """

    def fit(self, raw_signals: RawSignals):
        super().fit(raw_signals)
        lengths = [len(rs) for rs in raw_signals]
        q1, q3 = np.percentile(lengths, [25,75])
        iqr = q3 - q1
        self._lower_bound = q1 - 1.5 * iqr
        self._upper_bound = q3 + 1.5 * iqr

        return self

    def transform(self, raw_signals: RawSignals):
        self._check_fitted()

        new_signals = raw_signals.initialize_empty()
        for r_signal in raw_signals:
            r_sig_len = len(r_signal)
            if r_sig_len >= self._lower_bound and r_sig_len <= self._upper_bound:
                new_signals.append(deepcopy(r_signal))

        return new_signals
