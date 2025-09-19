import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter

from copy import deepcopy


class RawSignalsFilterColumnStandarizer(RawSignalsFilter):

    def _compute_means(self, raw_signals: RawSignals):
        dtype = raw_signals[0].signal.dtype
        n_channels = raw_signals[0].signal.shape[1]
        self._col_means = np.zeros(n_channels, dtype=dtype)
        self._n_samples = np.array(0, dtype=dtype)
        for r_sig in raw_signals:
            np_data = r_sig.to_numpy()
            self._col_means += np.sum(np_data, axis=0)
            self._n_samples += np_data.shape[0]

        self._col_means /= self._n_samples

        return self._col_means

    def _compute_stds(self, raw_signals:RawSignals, eps=1e-8):
        dtype = raw_signals[0].signal.dtype
        n_channels = raw_signals[0].signal.shape[1]
        self._stds = np.zeros(n_channels, dtype=dtype)

        for r_sig in raw_signals:
            np_data = r_sig.to_numpy()
            self._stds += np.sum( (np_data - self._col_means)**2,axis=0)

        self._stds = np.sqrt(self._stds / self._n_samples)
        self._stds = np.where(self._stds<eps,1,self._stds)
        return self._stds

    def fit(self, raw_signals: RawSignals):
        self._compute_means(raw_signals)
        self._compute_stds(raw_signals)
        self._fitted = True
        return self

    def transform(self, raw_signals: RawSignals):

        if not hasattr(self, '_fitted'):
            raise RuntimeError("Filter not fitted. Call 'fit' with training data before using this method.")

        copied_signals = deepcopy(raw_signals)
        for r_signal in copied_signals:
            r_signal.signal = (r_signal.signal - self._col_means)/self._stds

        return copied_signals
