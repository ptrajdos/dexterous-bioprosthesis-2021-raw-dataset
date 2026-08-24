from scipy.interpolate import interp1d
from copy import deepcopy
import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signal_augumenter_base import (
    RawSignalsAugumenterBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal


class RawSignalsAugumenterTimeJiter(RawSignalsAugumenterBase):

    def __init__(
        self,
        jiter_std=0.05,
        interpolation="cubic",
        independent_channels=False,
        n_repeats: int = 2,
        append_original=True,
        n_jobs=None,
        random_state=10,
    ) -> None:
        super().__init__(
            n_jobs=n_jobs,
            append_original=append_original,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        self.jiter_std = jiter_std
        self.interpolation = interpolation
        self.independant_channels = independent_channels

    def _sig_augument(self, raw_signal: RawSignal, n_repeats: int = 1):
        sig_list = []

        for _ in range(n_repeats):
            new_signal = deepcopy(raw_signal)
            np_sig = new_signal.signal
            n_samples, n_channels = np_sig.shape
            dt = 1 / new_signal.sample_rate
            t = np.arange(0, n_samples * dt, dt)
            jiter_std = self.jiter_std * dt

            time_jiter = self._random_state.normal(0, jiter_std, n_samples)
            interpolator = interp1d(
                time_jiter,
                np_sig,
                kind=self.interpolation,
                axis=0,
                fill_value="extrapolate",
            )
            np_sig = interpolator(t)
            
            sig_list.append(new_signal)

        return sig_list
