import warnings

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_energy_over_entropy import (
    NpSignalExtractorEnergyOverEntropy,
)
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import (
    NpSignalExtractorTest,
)


class NpSignalExtractorEnergyOverEntropyTest(NpSignalExtractorTest):

    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorEnergyOverEntropy(),
            NpSignalExtractorEnergyOverEntropy(
                sanitize_output=True, check_input=True, check_output=True
            ),
        ]

    def test_constant_signal(self):
        fs = 500
        X = np.ones((1000, 2))
        res = NpSignalExtractorEnergyOverEntropy().fit_transform(X, fs=fs)
        self.assertTrue(np.allclose(res, (1000 / fs) / np.log2(1000)))

    def test_fs_scaling(self):
        X = np.random.random((200, 3))
        res_1k = NpSignalExtractorEnergyOverEntropy().fit_transform(X, fs=1000)
        res_2k = NpSignalExtractorEnergyOverEntropy().fit_transform(X, fs=2000)
        self.assertTrue(np.allclose(res_1k, 2 * res_2k))

    def test_zero_samples_no_warnings(self):
        X = np.zeros((100, 2))
        X[:, 1] = 1
        X[5, 1] = 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(all="raise"):
                res = NpSignalExtractorEnergyOverEntropy().fit_transform(X)
        self.assertEqual(res[0], 0.0)
        self.assertTrue(np.isclose(res[1], (99 / 1000) / np.log2(99)))

    def test_single_spike(self):
        X = np.zeros((100, 1))
        X[3, 0] = 2
        res = NpSignalExtractorEnergyOverEntropy().fit_transform(X)
        self.assertEqual(res[0], 0.0)

    def test_int16_input(self):
        X = np.full((100, 1), 300, dtype=np.int16)
        res = NpSignalExtractorEnergyOverEntropy().fit_transform(X, fs=1000)
        expected = (300.0 ** 2 * 100 / 1000) / np.log2(100)
        self.assertTrue(np.issubdtype(res.dtype, np.floating))
        self.assertTrue(np.isclose(res[0], expected))
