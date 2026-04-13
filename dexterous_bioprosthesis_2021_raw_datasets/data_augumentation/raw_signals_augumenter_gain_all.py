from copy import deepcopy


from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signal_augumenter_base import (
    RawSignalsAugumenterBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal


class RawSignalsAugumenterGainAll(RawSignalsAugumenterBase):

    def __init__(
        self,
        gain_perc_min=0.01,
        gain_perc_max=2.0,
        n_repeats: int = 2,
        append_original=True,
        n_jobs=None,
        random_state=10,
    ) -> None:
        super().__init__(
            n_jobs=n_jobs, append_original=append_original, n_repeats=n_repeats, random_state=random_state,
        )

        self.gain_perc_min = gain_perc_min
        self.gain_perc_max = gain_perc_max

    def _sig_augument(self, raw_signal: RawSignal, n_repeats: int = 1):
        sig_list = []

        for _ in range(n_repeats):
            new_signal = deepcopy(raw_signal)
            np_sig = new_signal.signal

            gain_perc = self._random_state.uniform(self.gain_perc_min, self.gain_perc_max, 1)
            np_sig *= gain_perc

            sig_list.append(new_signal)

        return sig_list
