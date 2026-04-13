import abc

from joblib import delayed
import numpy as np
from sklearn.utils import check_random_state
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.tools.progressparallel import (
    ProgressParallel,
)


class RawSignalsAugumenterBase(RawSignalsAugumenter):

    def __init__(self, n_jobs=None, append_original=True, n_repeats:int=1, random_state=10) -> None:
        super().__init__()
        self.n_jobs = n_jobs
        self.append_original:bool = append_original
        self.n_repeats:int = n_repeats
        self.random_state = random_state

    @abc.abstractmethod
    def _sig_augument(self, raw_signal: RawSignal, n_repeats: int=1) -> list:
        """
        Auguments a single signal

        Arguments:
        ---------
        raw_signal: RawSignal -- the signal to be augumented
        n_repeats: int -- how many augumented versions of signal to create

        Returns:
        --------
        List of augumented signals

        """
    
    def _check_if_fitted(self):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        self._check_if_fitted()
        new_signals = raw_signals.initialize_empty()

        aug_sig_list = ProgressParallel(
            n_jobs=self.n_jobs, use_tqdm=True, total=len(raw_signals)
        )(delayed(self._sig_augument)(sig, self.n_repeats) for sig in raw_signals)

        for aug_sigs in aug_sig_list:
            new_signals += aug_sigs

        if self.append_original:
            new_signals += raw_signals

        return new_signals

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        self.fit(raw_signals)
        return self.transform(raw_signals)
    
    def fit(self, raw_signals: RawSignals) -> RawSignalsAugumenter:
        self._is_fitted = True
        self._random_state = check_random_state(self.random_state)
        return self
    
    def sample(self, raw_signals: RawSignals, n_samples: int=1) -> RawSignals:
        self._check_if_fitted()
        n_signals = len(raw_signals)
        replace = n_samples > n_signals
        indices = np.random.choice(len(raw_signals), size=n_samples, replace=replace)
        new_signals = raw_signals.initialize_empty()
        sel_signals = raw_signals[indices]

        for sig in sel_signals:
            new_signals += self._sig_augument(sig, n_repeats=1)
            
        return new_signals
