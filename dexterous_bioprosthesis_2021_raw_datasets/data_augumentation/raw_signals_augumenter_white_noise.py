from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signal_augumenter_base import (
    RawSignalsAugumenterBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsAugumenterWhiteNoise(RawSignalsAugumenterBase):

    def __init__(
        self,
        noise_perc_min=0.01,
        noise_perc_max=1.0,
        n_repeats: int = 2,
        append_original=True,
        n_jobs=None,
    ) -> None:
        super().__init__(
            n_jobs=n_jobs, append_original=append_original, n_repeats=n_repeats
        )

        self.noise_perc_min = noise_perc_min
        self.noise_perc_max = noise_perc_max

    def _sig_augument(self, raw_signal: RawSignal, n_repeats: int = 1) -> list:
        sig_list = []

        orig_sig = raw_signal.signal
        n_samples, n_channels = orig_sig.shape

        for _ in range(n_repeats):
            new_signal = deepcopy(raw_signal)

            noise_perc = np.random.uniform(
                self.noise_perc_min, self.noise_perc_max, (1, n_channels)
            )
            stds = orig_sig.std(axis=0, keepdims=True)  # shape (1, n_channels)
            noise = np.random.normal(0, 1, (n_samples, n_channels)) * stds
            new_signal.signal += noise_perc * noise
            sig_list.append(new_signal)

        return sig_list
