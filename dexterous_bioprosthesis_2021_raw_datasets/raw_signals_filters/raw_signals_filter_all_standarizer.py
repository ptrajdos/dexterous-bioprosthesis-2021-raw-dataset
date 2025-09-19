import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

from copy import deepcopy


class RawSignalsFilterAllStandarizer(RawSignalsFilter):

    def _compute_mean(self, raw_signals: RawSignals):
        dtype = raw_signals[0].signal.dtype
        self._glob_mean = np.array(0, dtype=dtype)
        self._n_samples = np.array(0, dtype=dtype)
        for r_sig in raw_signals:
            np_data = r_sig.to_numpy()
            self._glob_mean += np.sum(np_data)
            self._n_samples += np_data.size

        self._glob_mean /= self._n_samples

        return self._glob_mean

    def _compute_std(self, raw_signals:RawSignals, eps=1e-8):
        dtype = raw_signals[0].signal.dtype
        self._std = np.array(0, dtype=dtype)

        for r_sig in raw_signals:
            np_data = r_sig.to_numpy()
            self._std += np.sum( (np_data - self._glob_mean)**2)

        self._std = np.sqrt(self._std / self._n_samples)
        self._std = np.where(self._std<eps,1,self._std)
        return self._std

    def fit(self, raw_signals: RawSignals):
        self._compute_mean(raw_signals)
        self._compute_std(raw_signals)
        self._fitted = True
        return self

    def transform(self, raw_signals: RawSignals):
        if not hasattr(self, '_fitted'):
            raise RuntimeError("Filter not fitted. Call 'fit' with training data before using this method.")

        copied_signals = deepcopy(raw_signals)
        for r_signal in copied_signals:
            r_signal.signal = (r_signal.signal - self._glob_mean)/self._std

        return copied_signals
