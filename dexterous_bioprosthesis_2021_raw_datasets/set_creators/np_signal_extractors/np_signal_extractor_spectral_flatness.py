"""Module implementing spectral flatness extraction.

Computes the spectral flatness (Wiener entropy) from the PSD.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral import (
    NpSignalExtractorSpectral,
)

import numpy as np



class NpSignalExtractorSpectralFlatness(NpSignalExtractorSpectral):
    """Extractor that computes the spectral flatness (Wiener entropy)."""

    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        return super().fit(X)

    def _transform(self, X):

        psd, freqs = NpSignalExtractorSpectral._calculate_psd(X=X)
        eps = 1e-12

        spectral_flatness = np.exp(np.mean(np.log(psd + eps),axis=0)) / (np.mean(psd,axis=0) + eps)
        spectral_flatness = spectral_flatness.astype(X.dtype)
        return spectral_flatness

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
