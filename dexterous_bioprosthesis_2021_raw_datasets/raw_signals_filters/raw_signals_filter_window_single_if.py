"""Module implementing single-window signal extraction.

Extracts a single window of specified offset and length from each signal.
Supports integer (absolute sample indices), float (fractional, 0-1), and mixed parameters.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterWindowSingleIF(RawSignalsFilter):
    """Filter that extracts a single window from each signal."""

    def __init__(self, offset: Real, length: Real | None) -> None:
        """Cuts single window from given RawSignals.

        Arguments:
        ---------
        offset:Real -- offset from the beginning of the signal (samples if int, fraction if float)
        length:Real|None -- length of the window (samples if int, fraction if float, None for rest of signal)

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

        if self.length is not None and not isinstance(self.length, Real):
            raise ValueError(
                f"length is not a numeric type. Got: {type(self.length)}"
            )

        effective_offset = self._resolve_param(
            self.offset, "offset", signal_length, is_offset=True
        )

        if self.length is None:
            effective_length = signal_length - effective_offset
        else:
            effective_length = self._resolve_param(
                self.length, "length", signal_length, is_offset=False
            )

        if effective_offset + effective_length > signal_length:
            raise ValueError(
                f"Effective offset + length ({effective_offset + effective_length}) is greater than signal length ({signal_length})"
            )

        return (effective_offset, effective_length)

    @staticmethod
    def _resolve_param(
        value: Real, name: str, signal_length: int, *, is_offset: bool
    ) -> int:
        """Resolves a single parameter to an absolute sample count.

        Arguments:
        ---------
        value:Real -- parameter value (int → absolute samples, float → fraction)
        name:str -- parameter name for error messages
        signal_length:int -- total signal length in samples
        is_offset:bool -- True for offset semantics, False for length semantics

        Returns:
        int -- resolved absolute value in samples

        """
        if isinstance(value, Integral):
            if is_offset:
                if value < 0:
                    raise ValueError(f"Offset is negative. Got: {value}")
                if value >= signal_length:
                    raise ValueError(
                        f"Offset ({value}) is greater than or equal to signal length ({signal_length})"
                    )
            else:
                if value < 1:
                    raise ValueError(f"Length is smaller than 1. Got: {value}")
            return int(value)

        if isinstance(value, Real):
            if is_offset:
                if not (value >= 0 and value < 1):
                    raise ValueError(
                        f"Offset should be within [0,1) interval. Got {value}"
                    )
                return int(round(value * signal_length))
            else:
                if not (value > 0 and value < 1):
                    raise ValueError(
                        f"Length should be within (0,1) interval. Got {value}"
                    )
                return max(int(round(value * signal_length)), 1)

        raise ValueError(
            f"{name} is not a numeric type. Got: {type(value)}"
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
