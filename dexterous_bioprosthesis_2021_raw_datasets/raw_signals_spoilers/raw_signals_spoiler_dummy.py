"""Module implementing a dummy (identity) spoiler.

Returns a deep copy of the input signals without any distortion.
"""
from copy import deepcopy

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler import (
    RawSignalsSpoiler,
)


class RawSignalsSpoilerDummy(RawSignalsSpoiler):

    """Identity spoiler that returns signals without distortion."""
    def __init__(
        self,
        channels_spoiled_frac=0.1,
        snr=1,
        random_state=10,
    ) -> None:
        super().__init__(channels_spoiled_frac, snr, random_state)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        self._check_is_fitted()
        return deepcopy(raw_signals)
