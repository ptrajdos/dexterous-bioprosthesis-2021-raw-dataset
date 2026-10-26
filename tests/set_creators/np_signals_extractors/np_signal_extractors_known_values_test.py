"""Known-value and edge-case tests for numpy signal extractors."""
import unittest
import warnings

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_complexity import (
    NpSignalExtractorComplexity,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_deltarms import (
    NpSignalExtractorDeltaRms,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_rms import (
    NpSignalExtractorRms,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_bandwidth import (
    NpSignalExtractorSpectralBandwidth,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_flatness import (
    NpSignalExtractorSpectralFlatness,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_kurtosis import (
    NpSignalExtractorSpectralKurtosis,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_moment import (
    NpSignalExtractorSpectralMoment,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_skew import (
    NpSignalExtractorSpectralSkew,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_temporal_kurtosis import (
    NpSignalExtractorTemporalKurtosis,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_temporal_moment import (
    NpSignalExtractorTemporalMoment,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_zcr import (
    NpSignalExtractorZcr,
)


def sine(freq, fs, n):
    return np.sin(2 * np.pi * freq * np.arange(n) / fs)[:, None]


class NpSignalExtractorsKnownValuesTest(unittest.TestCase):

    def test_spectral_centroid_in_hz(self):
        for fs in [500, 1000, 2000]:
            with self.subTest(fs=fs):
                res = NpSignalExtractorSpectralMoment().fit_transform(sine(100, fs, fs), fs=fs)
                self.assertTrue(np.isclose(res[0], 100.0))

    def test_spectral_bandwidth_pure_tone_is_zero(self):
        res = NpSignalExtractorSpectralBandwidth().fit_transform(sine(100, 1000, 1000), fs=1000)
        self.assertTrue(np.isclose(res[0], 0.0, atol=1e-6))

    def test_spectral_flatness_scale_invariant(self):
        X = np.random.default_rng(0).normal(size=(256, 1))
        ex = NpSignalExtractorSpectralFlatness()
        self.assertTrue(np.isclose(ex.fit_transform(X), ex.fit_transform(X * 1e-6)).all())

    def test_zero_signal_degenerate_values(self):
        X = np.zeros((100, 2))
        extractors = [
            NpSignalExtractorSpectralFlatness(),
            NpSignalExtractorSpectralKurtosis(),
            NpSignalExtractorSpectralSkew(),
            NpSignalExtractorSpectralMoment(),
            NpSignalExtractorSpectralBandwidth(),
            NpSignalExtractorTemporalKurtosis(),
            NpSignalExtractorComplexity(),
        ]
        for ex in extractors:
            with self.subTest(extractor=type(ex).__name__):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    res = ex.fit_transform(X)
                self.assertTrue(np.all(res == 0.0))

    def test_temporal_kurtosis_gaussian_envelope(self):
        t = np.linspace(0, 1, 1001)
        X = np.exp(-(((t - 0.5) / 0.05) ** 2) / 4)[:, None]  # energy is Gaussian in time
        res = NpSignalExtractorTemporalKurtosis().fit_transform(X)
        self.assertTrue(np.isclose(res[0], 0.0, atol=1e-6))

    def test_temporal_moment_small_amplitude(self):
        X = np.random.default_rng(0).normal(size=(200, 1))
        ex = NpSignalExtractorTemporalMoment()
        self.assertTrue(np.isclose(ex.fit_transform(X), ex.fit_transform(X * 1e-6)).all())

    def test_temporal_moment_seconds(self):
        X = np.ones((1001, 1))
        for fs in [500, 1000]:
            with self.subTest(fs=fs):
                res = NpSignalExtractorTemporalMoment(proportional_time=False).fit_transform(X, fs=fs)
                self.assertTrue(np.isclose(res[0], 500 / fs))

    def test_zcr_rate(self):
        res = NpSignalExtractorZcr().fit_transform(sine(100, 1000, 1000))
        self.assertTrue(np.isclose(res[0], 0.2, atol=0.005))

    def test_int16_input(self):
        X = np.full((10, 1), 300, dtype=np.int16)
        res = NpSignalExtractorRms().fit_transform(X)
        self.assertTrue(np.isclose(res[0], 300.0))

    def test_deltarms_uses_fit_fs(self):
        X = np.random.default_rng(0).normal(size=(1000, 1))
        ex = NpSignalExtractorDeltaRms(window_length_ms=10, offset_ms=5)
        res_1k = ex.fit(X, fs=1000).transform(X)
        res_100 = ex.fit(X, fs=100).transform(X)
        self.assertFalse(np.isclose(res_1k, res_100).all())


if __name__ == "__main__":
    unittest.main()
