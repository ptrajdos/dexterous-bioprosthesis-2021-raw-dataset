import warnings

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_energy_entropy import NpSignalExtractorEnergyEntropy
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mobility import (
    NpSignalExtractorMobility,
)
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import (
    NpSignalExtractorTest,
)


class NpSignalExtractorEnergyEntropyTest(NpSignalExtractorTest):

    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorEnergyEntropy(),
            NpSignalExtractorEnergyEntropy(
                sanitize_output=True, check_input=True, check_output=True
            ),
        ]

    def test_constant_signal(self):
        X = np.ones((1000, 2))
        res = NpSignalExtractorEnergyEntropy().fit_transform(X)
        self.assertTrue(np.allclose(res, np.log2(1000)))

    def test_zero_channel(self):
        X = np.zeros((100, 2))
        X[:, 1] = np.linspace(1, 2, 100)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = NpSignalExtractorEnergyEntropy().fit_transform(X)
        self.assertEqual(res[0], 0.0)
        self.assertFalse(np.signbit(res[0]))
        self.assertTrue(res[1] > 0)

    def test_integer_input(self):
        X = np.ones((16, 3), dtype=int)
        res = NpSignalExtractorEnergyEntropy().fit_transform(X)
        self.assertTrue(np.issubdtype(res.dtype, np.floating))
        self.assertTrue(np.allclose(res, 4.0))

    def test_fs_forwarded(self):
        extractor = NpSignalExtractorEnergyEntropy().fit(np.ones((10, 1)), fs=250)
        self.assertEqual(extractor._fs, 250)
