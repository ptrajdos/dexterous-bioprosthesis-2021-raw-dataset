from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral import (
    NpSignalExtractorSpectral,
)

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_moment import (
    NpSignalExtractorSpectralMoment,
)


class NpSignalExtractorSpectralKurtosis(NpSignalExtractorSpectral):

    def fit(self, X):
        return super().fit(X)

    def _transform(self, X):

        psd, freqs = NpSignalExtractorSpectral._calculate_psd(X=X)

        u4 = NpSignalExtractorSpectralMoment._spectral_moment(psd=psd,freqs=freqs, order=4, centered=True)
        u2 = NpSignalExtractorSpectralMoment._spectral_moment(psd=psd,freqs=freqs, order=2, centered=True)

        kurtosis = u4 / (u2**2) - 3.0
        kurtosis = kurtosis.astype(X.dtype)

        return kurtosis
        

    def attribs_per_column(self):
        return 1
