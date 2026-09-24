"""Module implementing per-column robust standardisation.

Standardises each channel independently using the median and IQR
computed per column across the dataset.
"""

from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterQuantileStandarizer(RawSignalsFilter):
    """Filter that applies per-column quantile-based standardisation."""

    def __init__(
        self,
        q_low: float = 0.01,
        q_high: float = 0.99,
        range_min: float = 0,
        range_max: float = 1,
        clip:bool = False,
        eps=1e-30,
    ) -> None:
        super().__init__()

        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.range_min = range_min
        self.range_max = range_max
        self.clip = clip
        self.eps = eps

    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        np_data = raw_signals.to_numpy_concat()

        self._q_low, self._q_high = np.percentile(
            np_data,
            [self.q_low * 100.0, self.q_high * 100.0],
            axis=0,
        )

        self._scale = self._q_high - self._q_low

        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_fitted()

        copied_signals = deepcopy(raw_signals)

        for r_signal in copied_signals:
            sig_dtype = r_signal.to_numpy().dtype

            r_signal.signal = (
                ((r_signal.signal - self._q_low) / (self._scale + self.eps))
                * (self.range_max - self.range_min)
                + self.range_min
            ).astype(sig_dtype)
            if self.clip:
                r_signal.signal = np.clip(r_signal.signal,self.range_min, self.range_max)

        return copied_signals
