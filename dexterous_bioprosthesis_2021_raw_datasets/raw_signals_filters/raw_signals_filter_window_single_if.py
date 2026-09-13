"""Module implementing single-window signal extraction.

Extracts a single window of specified offset and length from each signal.
Supports both integer (absolute sample indices) and float (fractional, 0-1) parameters.
"""
from copy import deepcopy
from numbers import Integral, Real

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterWindowSingleIF(RawSignalsFilter):
    """Filter that extracts a single window from each signal."""

    def __init__(self, offset: Real, length: Real) -> None:
        """Cuts single window from given RawSignals.

        Arguments:
        ---------
        offset:Real -- offset from the beginning of the signal (samples if int, fraction if float)
        length:Real -- length of the window (samples if int, fraction if float)

        """
        super().__init__()
        self.offset = offset
        self.length = length

    def _get_effective_offset_and_length(
        self, signal_length: int
    ) -> tuple[Integral, Integral]:
        """Calculates effective offset and length based on signal length.

        Arguments:
        ---------
        signal_length:int -- length of the signal in samples

        Returns:
        effective_offset:int -- effective offset in samples
        effective_length:int -- effective length in samples

        """
        if not isinstance(self.offset, Real):
            raise ValueError(
                f"offset is not a numeric type. Got: {type(self.offset)}"
            )

        if not isinstance(self.length, Real):
            raise ValueError(
                f"length is not a numeric type. Got: {type(self.length)}"
            )

        if isinstance(self.offset, Integral) and isinstance(self.length, Integral):
            if self.offset < 0:
                raise ValueError(f"Offset is negative. Got: {self.offset}")

            if self.length < 1:
                raise ValueError(f"Length is smaller than 1. Got: {self.length}")

            if self.offset >= signal_length:
                raise ValueError(
                    f"Offset ({self.offset}) is greater than or equal to signal length ({signal_length})"
                )

            if self.offset + self.length > signal_length:
                raise ValueError(
                    f"Offset + length ({self.offset + self.length}) is greater than signal length ({signal_length})"
                )

            return (self.offset, self.length)

        if isinstance(self.offset, Real) and isinstance(self.length, Real):
            if not (self.offset >= 0 and self.offset < 1):
                raise ValueError(
                    f"Offset should be within [0,1) interval. Got {self.offset}"
                )

            if not (self.length > 0 and self.length < 1):
                raise ValueError(
                    f"Length should be within (0,1) interval. Got {self.length}"
                )

            if self.offset + self.length > 1:
                raise ValueError(
                    f"Offset + length ({self.offset + self.length}) exceeds 1.0"
                )

            effective_offset = int(round(self.offset * signal_length))
            effective_length = max(int(round(self.length * signal_length)), 1)

            if effective_offset + effective_length > signal_length:
                raise ValueError(
                    f"Effective offset + length ({effective_offset + effective_length}) is greater than signal length ({signal_length})"
                )

            return (effective_offset, effective_length)

        raise ValueError(
            f"offset and length must be both integers or both floats. Got: {type(self.offset)} and {type(self.length)}"
        )

    def fit(self, raw_signals: RawSignals, y=None):
        """Does nothing
        """
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Extracts a window from each signal based on offset and length.
        """
        self._check_fitted()
        copied_signals = deepcopy(raw_signals)

        for signal in copied_signals:
            s_len = signal.signal.shape[0]
            effective_offset, effective_length = self._get_effective_offset_and_length(s_len)
            start_sample = effective_offset
            end_sample = effective_offset + effective_length
            signal.signal = signal.signal[start_sample:end_sample, :]

        return copied_signals
