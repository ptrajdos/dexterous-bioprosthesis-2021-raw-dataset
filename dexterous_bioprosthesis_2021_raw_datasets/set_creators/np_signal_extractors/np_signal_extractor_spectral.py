from abc import ABC

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np

class NpSignalExtractorSpectral(NPSignalExtractor, ABC):

    @staticmethod
    def _calculate_psd(X, fs=1.0):
        """Calculate the Power Spectral Density (PSD) of the signal using FFT.

        Parameters:
        X (np.ndarray): Input signal of shape (n_samples, n_channels).

        Returns:
        psd (np.ndarray): Power Spectral Density of shape (n_freqs, n_channels).
        freqs (np.ndarray): Corresponding frequency bins of shape (n_freqs,).
        """
        n_samples = X.shape[0]
        fft_vals = np.fft.rfft(X, axis=0)
        fft_freqs = np.fft.rfftfreq(n_samples, d=1/fs)
        psd = (1.0 / (n_samples*fs)) * np.abs(fft_vals) ** 2
        return psd, fft_freqs