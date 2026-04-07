import abc
from copy import deepcopy

from joblib import delayed
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.tools.progressparallel import (
    ProgressParallel,
)


class RawSignalsAugumenterBase(RawSignalsAugumenter):

    def __init__(self, n_jobs=None, append_original=True) -> None:
        super().__init__()
        self.n_jobs = n_jobs
        self.append_original = append_original

    @abc.abstractmethod
    def _sig_augument(self, raw_signal: RawSignal) -> list:
        """
        Auguments a single signal

        Arguments:
        ---------
        raw_signal: RawSignal -- the signal to be augumented

        Returns:
        --------
        List of augumented signals

        """

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        new_signals = raw_signals.initialize_empty()

        aug_sig_list = ProgressParallel(
            n_jobs=self.n_jobs, use_tqdm=True, total=len(raw_signals)
        )(delayed(self._sig_augument)(sig) for sig in raw_signals)

        for aug_sigs in aug_sig_list:
            new_signals += aug_sigs

        if self.append_original:
            new_signals += raw_signals

        return new_signals

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        self.fit(raw_signals)
        return self.transform(raw_signals)
