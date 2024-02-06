from joblib import delayed
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import RawSignalsAugumenter
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from copy import deepcopy
import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.tools.progressparallel import ProgressParallel

class RawSignalsAugumenterInvertPolarity(RawSignalsAugumenter):

    def __init__(self, append_original = True, n_jobs=None) -> None:
        super().__init__()
        self.n_jobs = n_jobs
        self.append_original = append_original


    def fit(self, raw_signals: RawSignals):
        """
        Intentionally does nothing
        """
        return self

    def _sig_augument(self, signal:RawSignal):
        new_signal = deepcopy(signal)
        np_sig = new_signal.signal
    
        np_sig*= -1.0
        
        return [new_signal]

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        new_signals = RawSignals()
        
        aug_sig_list = ProgressParallel(n_jobs=self.n_jobs,use_tqdm=True,total=len(raw_signals))(delayed(self._sig_augument)( sig ) for sig in raw_signals  )

        for aug_sigs in aug_sig_list:
            new_signals+=aug_sigs    

        if self.append_original:
            new_signals+= raw_signals

        return new_signals

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        self.fit(raw_signals)
        return self.transform(raw_signals)
        