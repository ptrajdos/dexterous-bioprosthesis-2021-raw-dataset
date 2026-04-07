from copy import deepcopy


from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signal_augumenter_base import (
    RawSignalsAugumenterBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsAugumenterInvertPolarity(RawSignalsAugumenterBase):

    def __init__(self, append_original=True, n_jobs=None) -> None:
        super().__init__(n_jobs=n_jobs, append_original=append_original)

    def fit(self, raw_signals: RawSignals):
        """
        Intentionally does nothing
        """
        return self

    def _sig_augument(self, raw_signal: RawSignal):
        new_signal = deepcopy(raw_signal)
        np_sig = new_signal.signal

        np_sig *= -1.0

        return [new_signal]
