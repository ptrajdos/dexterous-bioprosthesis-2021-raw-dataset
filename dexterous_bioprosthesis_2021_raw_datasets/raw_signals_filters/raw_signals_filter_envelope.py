"""Module implementing envelope extraction via the Hilbert transform.

Computes the analytic signal envelope of each channel using
:func:`scipy.signal.hilbert`.
"""
from copy import deepcopy

import numpy as np
from scipy.signal import hilbert

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterEnvelope(RawSignalsFilter):

    """Filter that extracts the signal envelope via the Hilbert transform."""
    def __init__(self) -> None:
        super().__init__()

    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):

        """Transform the given data."""
        self._check_fitted()

        copied_signals = deepcopy(raw_signals)

        for raw_signal in copied_signals:
            sig_np = raw_signal.signal
            n_ch = sig_np.shape[1]

            for ch_idx in range(n_ch):
                an_sig = hilbert(sig_np[:, ch_idx])
                sig_np[:, ch_idx] = np.abs(an_sig)

        return copied_signals
