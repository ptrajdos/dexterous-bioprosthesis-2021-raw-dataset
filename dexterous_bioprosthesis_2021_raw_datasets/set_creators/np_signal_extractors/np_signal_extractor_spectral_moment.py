from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral import (
    NpSignalExtractorSpectral,
)

import numpy as np


class NpSignalExtractorSpectralMoment(NpSignalExtractorSpectral):

    def __init__(
        self,
        sanitize_output=False,
        check_input=False,
        check_output=False,
        order=1,
        centered=True,
    ):
        super().__init__(sanitize_output, check_input, check_output)
        self.order = order
        self.centered = centered

    def fit(self, X, fs=1000):
        return super().fit(X)

    @staticmethod
    def _spectral_centroid(psd, freqs):
        spectral_centroid = np.sum(freqs[:, np.newaxis] * psd, axis=0) / np.sum(
            psd, axis=0
        )
        return spectral_centroid

    @staticmethod
    def _spectral_moment(psd, freqs, order, centered):
        spectral_centroid = 0
        if centered:
            spectral_centroid = NpSignalExtractorSpectralMoment._spectral_centroid(psd=psd, freqs=freqs)

        moment = np.sum( psd*(freqs[:,np.newaxis] - spectral_centroid) ** order,axis=0 )/np.sum(psd,axis=0)
        return moment

    def _transform(self, X):

        psd, freqs = NpSignalExtractorSpectral._calculate_psd(X=X)
        if self.order == 1:
            return NpSignalExtractorSpectralMoment._spectral_centroid(
                psd=psd, freqs=freqs
            ).astype(X.dtype)

        return NpSignalExtractorSpectralMoment._spectral_moment(
            psd=psd, freqs=freqs, order=self.order, centered=self.centered
        ).astype(X.dtype)

    def attribs_per_column(self):
        return 1
