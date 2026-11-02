from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_offsetratio import (
    NpSignalExtractorOffsetRatio,
)
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import (
    NpSignalExtractorTest,
)

import numpy as np


class NpSignalExtractorOffsetRatioTest(NpSignalExtractorTest):

    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorOffsetRatio(),
            NpSignalExtractorOffsetRatio(
                sanitize_output=True, check_input=True, check_output=True
            ),
        ]

    def test_known_values(self):
        # Columns: symmetric zero-mean, positive constant, negative constant,
        # zeros, and a signal with mean 1 and mean(|x|) 2.
        X = np.array(
            [
                [1.0, 2.0, -3.0, 0.0, 3.0],
                [-1.0, 2.0, -3.0, 0.0, -1.0],
                [1.0, 2.0, -3.0, 0.0, 3.0],
                [-1.0, 2.0, -3.0, 0.0, -1.0],
            ]
        )
        expected = np.array([0.0, 1.0, 1.0, 0.0, 0.5])

        for extractor in self.get_extractors():
            with self.subTest(extractor=extractor):
                T = extractor.fit_transform(X)
                np.testing.assert_allclose(T, expected, atol=1e-9)

    def test_range(self):
        X = np.random.random((200, 4)) - 0.3

        for extractor in self.get_extractors():
            with self.subTest(extractor=extractor):
                T = extractor.fit_transform(X)
                self.assertTrue(np.all(T >= 0.0), "Negative values in output")
                self.assertTrue(np.all(T <= 1.0), "Values greater than one in output")

    def test_amplitude_invariance(self):
        X = np.random.random((200, 3)) - 0.3

        for extractor in self.get_extractors():
            with self.subTest(extractor=extractor):
                T = extractor.fit_transform(X)
                for scale in [1e-3, 0.5, 10.0, 1e3]:
                    T_scaled = extractor.fit_transform(X * scale)
                    np.testing.assert_allclose(T_scaled, T, rtol=1e-6)

    def test_sign_invariance(self):
        X = np.random.random((200, 3)) - 0.3

        for extractor in self.get_extractors():
            with self.subTest(extractor=extractor):
                np.testing.assert_allclose(
                    extractor.fit_transform(-X), extractor.fit_transform(X)
                )

    def test_offset_increases_ratio(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1000, 1))
        offsets = [0.0, 0.5, 1.0, 2.0, 5.0]

        extractor = NpSignalExtractorOffsetRatio()
        ratios = [extractor.fit_transform(X + o)[0] for o in offsets]
        self.assertTrue(
            np.all(np.diff(ratios) > 0),
            f"Ratio should grow with baseline drift, got {ratios}",
        )

    def test_attribute_name(self):
        self.assertEqual(
            NpSignalExtractorOffsetRatio().get_attribute_names(), ["offset_ratio"]
        )
