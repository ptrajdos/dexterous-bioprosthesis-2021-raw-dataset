import itertools
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import (
    SetCreator,
)
from sklearn.exceptions import NotFittedError
import numpy as np
import pywt
import abc
from sktime.libs.vmdpy import VMD


class SetCreatorVMD(SetCreator):

    def __init__(
        self, alpha=2000, tau=0.0, K=3, DC=0, init=1, tol=1e-7, extractors=[]
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol
        self.extractors = extractors

        self._num_attribs = None
        self.n_channels = None
        self.channel_selected_attribs = (
            None  # List containing number of attributes for each channel
        )

    def fit(self, raw_signals: RawSignals, y=None):
        self.n_channels = raw_signals[0].to_numpy().shape[1]
        n_extractors = len(self.extractors)
        n_levels = self.K

        self.channel_selected_attribs = []
        for ch_id in range(self.n_channels):
            self.channel_selected_attribs.append([])

        offset = 0
        for extr_id in range(n_extractors):
            n_attribs_per_channel = self.extractors[extr_id].attribs_per_column()
            for level_id in range(n_levels):
                for ch_id in range(self.n_channels):
                    tmp_attrib_idxs = [offset + i for i in range(n_attribs_per_channel)]
                    self.channel_selected_attribs[ch_id] += tmp_attrib_idxs
                    offset += n_attribs_per_channel

        self._num_attribs = offset

        return self

    def _decompose_signal(self, signal, fs=1000):
        """
        Generates decomposition coefficients of the signal.

        Returns
        list of tuples (decomposition coefficients, sampling frequency)
        """
        n_rows, n_cols = signal.shape

        tmp_array = np.zeros((self.K, n_rows, n_cols))

        for ch_idx in range(n_cols):
            tmp_signal = signal[:, ch_idx]
            u, u_hat, omega = VMD(
                tmp_signal,
                alpha=self.alpha,
                tau=self.tau,
                K=self.K,
                DC=self.DC,
                init=self.init,
                tol=self.tol,
            )
            tmp_array[:, :, ch_idx] = u

        freqs = itertools.repeat(fs, self.K)
        results = [(tmp_array[k], fs) for k, fs in enumerate(freqs)]

        return results

    def transform(self, raw_signals: RawSignals):

        if self.get_channel_attribs_indices() is None:
            raise NotFittedError("SetCreator has not been fitted.")

        n_signals = len(raw_signals)
        extracted_attribs = np.zeros((n_signals, self._num_attribs))
        labels = []
        timestamps = []

        for raw_signal_id, raw_signal in enumerate(raw_signals):

            signal = raw_signal.to_numpy()
            orig_fs = raw_signal.get_sample_rate()
            labels.append(raw_signal.get_label())
            timestamps.append(raw_signal.get_timestamp())
            decomposeds = self._decompose_signal(signal=signal, fs=orig_fs)
            offset = 0

            for extractor_id, extractor in enumerate(self.extractors):
                for decomposed_level, fs in decomposeds:
                    extracted = extractor.fit_transform(decomposed_level, fs=fs)
                    n_extracted = extracted.shape[0]
                    extracted_attribs[
                        raw_signal_id, offset : (offset + n_extracted)
                    ] = extracted
                    offset += n_extracted

        extracted_attribs = extracted_attribs.astype(raw_signals[0].to_numpy().dtype)
        labels = np.asanyarray(labels)
        timestamps = np.asanyarray(timestamps)
        return extracted_attribs, labels, timestamps

    def fit_transform(self, raw_signals: RawSignals, y=None):
        return self.fit(raw_signals).transform(raw_signals)

    def get_channel_attribs_indices(self):
        return self.channel_selected_attribs
