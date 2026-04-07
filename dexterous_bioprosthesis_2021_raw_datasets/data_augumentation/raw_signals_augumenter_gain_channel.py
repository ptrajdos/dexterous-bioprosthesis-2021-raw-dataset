from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signal_augumenter_base import (
    RawSignalsAugumenterBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsAugumenterGainChannel(RawSignalsAugumenterBase):

    def __init__(
        self,
        gain_perc_min=0.01,
        gain_perc_max=2.0,
        n_repeats: int = 2,
        append_original=True,
        n_jobs=None,
    ) -> None:
        super().__init__(n_jobs=n_jobs, append_original=append_original)

        self.gain_perc_min = gain_perc_min
        self.gain_perc_max = gain_perc_max
        self.n_repeats = n_repeats

    def fit(self, raw_signals: RawSignals):
        """
        Intentionally does nothing
        """
        return self

    def _sig_augument(self, raw_signal: RawSignal):
        sig_list = []

        for _ in range(self.n_repeats):
            new_signal = deepcopy(raw_signal)
            gain_perc = np.random.uniform(
                self.gain_perc_min, self.gain_perc_max, new_signal.signal.shape[1]
            )
            new_signal.signal *= gain_perc
            sig_list.append(new_signal)

        return sig_list
