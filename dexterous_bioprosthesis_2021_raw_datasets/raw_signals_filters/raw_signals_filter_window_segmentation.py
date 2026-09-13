"""Module implementing window-based signal segmentation.

Splits each signal into fixed-length overlapping or non-overlapping
window segments.
"""

from numbers import Real, Integral
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterWindowSegmentation(RawSignalsFilter):
    """Filter that segments signals into fixed-length windows."""

    def __init__(self, window_length: Real, overlap: Real) -> None:
        """Segment the signal using sliding window.
        New signals appear in the object

        Arguments:
        ---------
        window_length:Real --  window length in samples
        overlap:Real -- overlap in samples

        """
        super().__init__()
        self.window_length = window_length
        self.overlap = overlap

    def _get_effective_window_length_and_overlap(
        self, signal_length: int
    ) -> tuple[Integral, Integral]:
        """Calculates effective window length and overlap based on signal length

        Arguments:
        ---------
        signal_length:int --  length of the signal in samples

        Returns:
        effective_window_length:int -- effective window length in samples
        effective_overlap:int -- effective overlap in samples

        """
        pass
        if not isinstance(self.window_length, Real):
            raise ValueError(
                f"window_length is not a numeric type. Got: {type(self.window_length)}"
            )

        if not isinstance(self.overlap, Real):
            raise ValueError(f"overlap is not a numeric type. Got {type(self.overlap)}")

        if isinstance(self.window_length, Integral) and isinstance(
            self.overlap, Integral
        ):
            if not self.window_length >=1:
                raise ValueError("Window length is smaller than 1")
            

            if self.window_length > signal_length:
                raise ValueError(
                    f"Window length  ({self.window_length}) is greather than signal length ({signal_length})"
                )

            if not self.overlap>=1:
                raise ValueError("Overlap is smaller than 1")
            
            if self.overlap > signal_length:
                raise ValueError(
                    f"overlap ({self.overlap}) is greather than signal length ({signal_length})"
                )

            if self.overlap > self.window_length:
                raise ValueError(
                    f"overlap ({self.overlap}) is greather than window length ({self.window_length})"
                )

            return (self.window_length, self.overlap)

        if isinstance(self.window_length, Real) and isinstance(self.overlap, Real):

            if not (self.window_length>0 and self.window_length<1):
                raise ValueError(f"Window length should be within (0,1) interval. Got {self.window_length}")
            
            if not (self.overlap>0 and self.overlap<1):
                raise ValueError(f"Overlap should be within (0,1) interval. Got {self.overlap}")

            effective_window_length = max(int(round(self.window_length * signal_length)), 1)
            effective_overlap = max(int(round(self.overlap * effective_window_length)), 1)

            if effective_window_length > signal_length:
                raise ValueError(
                    f"Effective window length  ({effective_window_length}) is greather than signal length ({signal_length})"
                )

            if effective_overlap > effective_window_length:
                raise ValueError(
                    f"Effective overlap ({effective_overlap}) is greather than effective window length ({effective_window_length})"
                )

            return (effective_window_length, effective_overlap)

        raise ValueError(
            f"window_length and overlap must be both integers or both floats. Got: {type(self.window_length)} and {type(self.overlap)}"
        )

    def fit(self, raw_signals: RawSignals, y=None):
        """Does nothing"""
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Apply windowed segmentation with overlap"""
        self._check_fitted()
        new_signals = RawSignals(sample_rate=raw_signals.sample_rate)

        for signal in raw_signals:
            s_len = signal.signal.shape[0]
            effective_window_length, effective_overlap = self._get_effective_window_length_and_overlap(
                s_len
            )
            start_idx = 0
            end_idx = start_idx + effective_window_length
            while end_idx <= s_len:
                copied_signal = signal[start_idx:end_idx, :]
                copied_signal.signal = copied_signal.signal.copy()
                new_signals.append(copied_signal)

                start_idx += effective_window_length - effective_overlap
                end_idx += effective_window_length - effective_overlap

        return new_signals
