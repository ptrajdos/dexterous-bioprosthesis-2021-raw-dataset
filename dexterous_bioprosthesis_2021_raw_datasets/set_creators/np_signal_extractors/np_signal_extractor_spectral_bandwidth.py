from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral import (
    NpSignalExtractorSpectral,
)

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_moment import (
    NpSignalExtractorSpectralMoment,
)


class NpSignalExtractorSpectralBandwidth(NpSignalExtractorSpectral):

    def fit(self, X, fs=1000):
        return super().fit(X)

    def _transform(self, X):

        psd, freqs = NpSignalExtractorSpectral._calculate_psd(X=X)

        return np.sqrt(
            NpSignalExtractorSpectralMoment._spectral_moment(
                psd=psd, freqs=freqs, order=2, centered=True
            )
        ).astype(X.dtype)

    def attribs_per_column(self):
        return 1
