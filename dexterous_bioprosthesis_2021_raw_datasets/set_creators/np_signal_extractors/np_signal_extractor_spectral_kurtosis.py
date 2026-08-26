"""Module implementing spectral kurtosis extraction.

Computes the kurtosis of the PSD distribution.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral import (
    NpSignalExtractorSpectral,
)

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_moment import (
    NpSignalExtractorSpectralMoment,
)


class NpSignalExtractorSpectralKurtosis(NpSignalExtractorSpectral):

    """Extractor that computes the spectral kurtosis from the PSD."""
    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        return super().fit(X)

    def _transform(self, X):

        psd, freqs = NpSignalExtractorSpectral._calculate_psd(X=X)

        u4 = NpSignalExtractorSpectralMoment._spectral_moment(psd=psd,freqs=freqs, order=4, centered=True)
        u2 = NpSignalExtractorSpectralMoment._spectral_moment(psd=psd,freqs=freqs, order=2, centered=True)

        kurtosis = np.divide(
            u4,
            u2**2,
            out=np.zeros_like(u4),  # set kurtosis=0 where u2==0
            where=u2 > 0
        ) - 3.0  # subtract 3 for excess kurtosis

        kurtosis = kurtosis.astype(X.dtype)

        return kurtosis
        

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
