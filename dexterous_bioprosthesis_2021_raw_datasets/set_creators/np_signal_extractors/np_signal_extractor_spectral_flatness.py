"""Module implementing spectral flatness extraction.

Computes the spectral flatness (Wiener entropy) from the PSD.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral import (
    NpSignalExtractorSpectral,
)

import numpy as np



class NpSignalExtractorSpectralFlatness(NpSignalExtractorSpectral):
    """Extractor that computes the spectral flatness (Wiener entropy)."""

    def _transform(self, X):

        psd, freqs = NpSignalExtractorSpectral._calculate_psd(X=X, fs=self._get_fs())
        # Geometric mean is 0 if any bin is 0; computed in log domain otherwise.
        all_positive = np.all(psd > 0, axis=0)
        log_psd = np.log(psd, out=np.zeros_like(psd), where=psd > 0)
        geometric_mean = np.where(all_positive, np.exp(np.mean(log_psd, axis=0)), 0.0)
        arithmetic_mean = np.mean(psd, axis=0)
        spectral_flatness = np.divide(
            geometric_mean,
            arithmetic_mean,
            out=np.zeros_like(arithmetic_mean),
            where=arithmetic_mean > 0,
        )
        spectral_flatness = spectral_flatness.astype(X.dtype)
        return spectral_flatness

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
