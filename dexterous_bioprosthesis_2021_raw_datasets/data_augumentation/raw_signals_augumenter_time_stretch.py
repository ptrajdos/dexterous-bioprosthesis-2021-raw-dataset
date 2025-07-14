from joblib import delayed
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from copy import deepcopy
import numpy as np
from audiomentations import TimeStretch

from dexterous_bioprosthesis_2021_raw_datasets.tools.progressparallel import (
    ProgressParallel,
)


class RawSignalsAugumenterTimeStretch(RawSignalsAugumenter):

    def __init__(
        self,
        stretch_min=0.8,
        stretch_max=1.25,
        n_repeats: int = 2,
        append_original=True,
        n_jobs=None,
    ) -> None:
        super().__init__()

        self.stretch_min = stretch_min
        self.stretch_max = stretch_max
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.append_original = append_original

    def fit(self, raw_signals: RawSignals):
        """
        Intentionally does nothing
        """
        return self

    def _sig_augument(self, signal: RawSignal, sample_rate):
        sig_list = []
        transformer = TimeStretch(
            p=1.0,
            leave_length_unchanged=True,
            min_rate=self.stretch_min,
            max_rate=self.stretch_max,
        )
        for _ in range(self.n_repeats):
            new_signal = deepcopy(signal)
            np_sig = new_signal.signal

            for ch_id in range(np_sig.shape[1]):

                ch_sig = np_sig[:, ch_id]
                # FIXME  Woraround for  nummba issue with different dtypes
                # FIXME NotImplementedError: Failed in nopython mode pipeline (step: native lowering)
                np_sig[:, ch_id] = transformer(ch_sig.astype(np.double), sample_rate=sample_rate).astype(np_sig.dtype)

            sig_list.append(new_signal)
        return sig_list

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        new_signals = raw_signals.initialize_empty()

        sample_rate = raw_signals.sample_rate

        aug_sig_list = ProgressParallel(
            n_jobs=self.n_jobs, use_tqdm=True, total=len(raw_signals)
        )(delayed(self._sig_augument)(sig, sample_rate) for sig in raw_signals)

        for aug_sigs in aug_sig_list:
            new_signals += aug_sigs

        if self.append_original:
            new_signals += raw_signals
        return new_signals

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        self.fit(raw_signals)
        return self.transform(raw_signals)
