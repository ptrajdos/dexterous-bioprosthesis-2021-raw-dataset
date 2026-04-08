from copy import deepcopy

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterAllPass(RawSignalsFilter):

    def fit(self, raw_signals: RawSignals):
        """
        Does nothing
        """
        return super().fit(raw_signals)

    def transform(self, raw_signals: RawSignals):
        """
        Just make a deep copy of an object
        """
        self._check_fitted()
        return deepcopy(raw_signals)
