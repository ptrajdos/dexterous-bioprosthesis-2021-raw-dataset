from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import RawSignalsAugumenter
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from copy import deepcopy

class RawSignalsAugumenterDummy(RawSignalsAugumenter):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, raw_signals: RawSignals):
        """
        Intentionally does nothing
        """
        return self

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        return deepcopy(raw_signals)

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        self.fit(raw_signals)
        return self.transform(raw_signals)
        