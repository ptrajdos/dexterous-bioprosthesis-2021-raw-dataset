import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

from copy import deepcopy


class RawSignalsFilterAllStandarizer(RawSignalsFilter):

    def _compute_mean(self, raw_signals: RawSignals):
        n_channels = raw_signals[0].signal.shape[1]
        self._col_mean = 0.0
        self._n_samples = 0
        for r_sig in raw_signals:
            np_data = r_sig.to_numpy()
            self._col_mean += np.sum(np_data)
            self._n_samples += np_data.size

        self._col_mean /= self._n_samples

        return self._col_mean

    def _compute_std(self, raw_signals:RawSignals, eps=1e-8):
        n_channels = raw_signals[0].signal.shape[1]
        self._std = np.ones(n_channels, dtype=raw_signals[0].signal.dtype)

        for r_sig in raw_signals:
            np_data = r_sig.to_numpy()
            self._std += np.sum( (np_data - self._col_mean)**2)

        self._std = np.sqrt(self._std / self._n_samples)
        self._std = np.where(self._std<eps,1,self._std)
        return self._std

    def fit(self, raw_signals: RawSignals):
        self._compute_mean(raw_signals)
        self._compute_std(raw_signals)
        return self

    def transform(self, raw_signals: RawSignals):

        copied_signals = deepcopy(raw_signals)
        for r_signal in copied_signals:
            r_signal.signal = (r_signal.signal - self._col_mean)/self._std

        return copied_signals
