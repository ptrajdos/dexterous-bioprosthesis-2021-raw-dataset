import unittest

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ar import (
    NpSignalExtractorAr,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mav import (
    NpSignalExtractorMav,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ssc import (
    NpSignalExtractorSsc,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_functions import (
    SetCreatorFunctions,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_window_delta import (
    SetCreatorWindowDelta,
)
from tests.set_creators.set_creator_test import SetCreatorTest

import warnings

from tests.testing_tools import generate_sample_data


class SetCreatorWindowDeltaTest(SetCreatorTest):

    __test__ = True

    def get_creators(self):
        extractors = {
            # "default": SetCreatorWindowDelta(extractors=[
            #     NpSignalExtractorMav(),
            #     NpSignalExtractorSsc(),
            # ])
            "w_0.5;o_0.5": SetCreatorWindowDelta(
                extractors=[
                    NpSignalExtractorMav(),
                    NpSignalExtractorSsc(),
                ],
                window_length=0.5,
                overlap=0.5,
            ),
        }
        return extractors

    def test_creator_fit_transform_long_window(self):

        creators = self.get_creators()
        for (
            signal_number,
            column_number,
            samples_number,
            class_indices,
        ) in [(10,3,100,[0,1])]:
            for creator_name, creator in creators.items():
                with self.subTest(
                    signal_number=signal_number,
                    column_number=column_number,
                    samples_number=samples_number,
                    class_indices=class_indices,
                    creator=creator,
                ):

                    raw_set = self.generate_sample_data(
                        samples_number=samples_number,
                        signal_number=signal_number,
                        column_number=column_number,
                        class_indices=class_indices,
                    )
                    n_samples = len(raw_set)

                    X, y, t = creator.fit_transform(raw_set)

                    self.basic_test_check(raw_set, X, y, t)

    # --- Output shape tests ---

    def test_output_shape_matches_expected_features(self):
        """X columns should equal n_extractors * n_windows * n_channels."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav(), NpSignalExtractorSsc()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=5, column_number=3, samples_number=20,
        )
        X, y, t = creator.fit_transform(raw_set)
        # 2 extractors * 3 windows * 3 channels = 18
        self.assertEqual(X.shape, (5, 18))

    def test_output_rows_match_signal_count(self):
        """X rows should equal the number of input signals."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()],
            window_length=0.5, overlap=0.5,
        )
        for n_signals in [1, 3, 10]:
            with self.subTest(n_signals=n_signals):
                raw_set = self.generate_sample_data(
                    signal_number=n_signals, column_number=2, samples_number=20,
                )
                X, y, t = creator.fit_transform(raw_set)
                self.assertEqual(X.shape[0], n_signals)

    def test_y_length_matches_x_rows(self):
        """y and t length should match X rows."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=7, column_number=2, samples_number=20,
        )
        X, y, t = creator.fit_transform(raw_set)
        self.assertEqual(len(y), X.shape[0])
        self.assertEqual(len(t), X.shape[0])

    # --- Window configuration tests ---

    def test_different_window_overlap_configs(self):
        """Different window/overlap float configs should produce valid output."""
        configs = [
            (0.5, 0.5),
            (0.33, 0.5),
            (0.25, 0.5),
        ]
        for wl, ol in configs:
            with self.subTest(window_length=wl, overlap=ol):
                creator = SetCreatorWindowDelta(
                    extractors=[NpSignalExtractorMav()],
                    window_length=wl, overlap=ol,
                )
                raw_set = self.generate_sample_data(
                    signal_number=3, column_number=2, samples_number=30,
                )
                X, y, t = creator.fit_transform(raw_set)
                self.basic_test_check(raw_set, X, y, t)

    def test_n_windows_stored_after_fit(self):
        """After fit, n_windows should be set correctly."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=3, column_number=2, samples_number=20,
        )
        creator.fit(raw_set)
        # window=10, overlap=5, step=5 → windows at [0:10],[5:15],[10:20] = 3
        self.assertEqual(creator.n_windows, 3)

    def test_single_window_no_overlap(self):
        """When window covers entire signal, should produce 1 window."""
        # window_length close to 1.0 but within (0,1)
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()],
            window_length=0.99, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=2, column_number=2, samples_number=20,
        )
        X, y, t = creator.fit_transform(raw_set)
        # effective_window = round(0.99*20) = 20, covers entire signal → 1 window
        self.assertEqual(creator.n_windows, 1)
        # 1 extractor * 1 window * 2 channels = 2
        self.assertEqual(X.shape[1], 2)

    # --- Extractor tests ---

    def test_single_extractor(self):
        """Using a single extractor should produce correct feature count."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=3, column_number=3, samples_number=20,
        )
        X, y, t = creator.fit_transform(raw_set)
        # 1 extractor * 3 windows * 3 channels = 9
        self.assertEqual(X.shape[1], 9)

    def test_three_extractors(self):
        """Using three extractors should produce correct feature count."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav(), NpSignalExtractorSsc(), NpSignalExtractorMav()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=3, column_number=3, samples_number=20,
        )
        X, y, t = creator.fit_transform(raw_set)
        # 3 extractors * 3 windows * 3 channels = 27
        self.assertEqual(X.shape[1], 27)

    # --- Channel tests ---

    def test_single_channel(self):
        """Single channel signals should work correctly."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav(), NpSignalExtractorSsc()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=3, column_number=1, samples_number=20,
        )
        X, y, t = creator.fit_transform(raw_set)
        # 2 extractors * 3 windows * 1 channel = 6
        self.assertEqual(X.shape[1], 6)

    def test_many_channels(self):
        """Many channel signals should work correctly."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=2, column_number=8, samples_number=20,
        )
        X, y, t = creator.fit_transform(raw_set)
        # 1 extractor * 3 windows * 8 channels = 24
        self.assertEqual(X.shape[1], 24)

    # --- Channel attribute indices tests ---

    def test_channel_attribs_indices_length(self):
        """channel_attribs_indices should have one entry per channel."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=2, column_number=4, samples_number=20,
        )
        creator.fit(raw_set)
        indices = creator.get_channel_attribs_indices()
        self.assertEqual(len(indices), 4)

    def test_channel_attribs_indices_cover_all_features(self):
        """All feature indices should be covered by channel attrib indices."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav(), NpSignalExtractorSsc()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=2, column_number=3, samples_number=20,
        )
        creator.fit(raw_set)
        indices = creator.get_channel_attribs_indices()
        all_indices = sorted([idx for ch in indices for idx in ch])
        self.assertEqual(all_indices, list(range(18)))

    def test_channel_attribs_indices_no_overlap(self):
        """Channel attrib indices should not overlap between channels."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav(), NpSignalExtractorSsc()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=2, column_number=3, samples_number=20,
        )
        creator.fit(raw_set)
        indices = creator.get_channel_attribs_indices()
        all_indices = [idx for ch in indices for idx in ch]
        self.assertEqual(len(all_indices), len(set(all_indices)))

    # --- Label preservation tests ---

    def test_labels_preserved(self):
        """Output labels should match input signal labels."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=5, column_number=2, samples_number=20,
            class_indices=[0, 1],
        )
        input_labels = np.array([s.get_label() for s in raw_set])
        X, y, t = creator.fit_transform(raw_set)
        np.testing.assert_array_equal(y, input_labels)

    def test_timestamps_preserved(self):
        """Output timestamps should match input signal timestamps."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=5, column_number=2, samples_number=20,
        )
        input_timestamps = np.array([s.get_timestamp() for s in raw_set])
        X, y, t = creator.fit_transform(raw_set)
        np.testing.assert_array_equal(t, input_timestamps)

    # --- No NaN/Inf tests ---

    def test_no_nan_inf_in_output(self):
        """Output X should contain no NaN or Inf values."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav(), NpSignalExtractorSsc()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=5, column_number=3, samples_number=20,
        )
        X, y, t = creator.fit_transform(raw_set)
        self.assertFalse(np.isnan(X).any(), "NaN values in X")
        self.assertFalse(np.isinf(X).any(), "Inf values in X")

    # --- Fit then transform consistency ---

    def test_fit_then_transform_same_as_fit_transform(self):
        """fit().transform() should produce same shape as fit_transform()."""
        creator1 = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav(), NpSignalExtractorSsc()],
            window_length=0.5, overlap=0.5,
        )
        creator2 = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav(), NpSignalExtractorSsc()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=5, column_number=3, samples_number=20,
        )
        X1, y1, t1 = creator1.fit_transform(raw_set)
        creator2.fit(raw_set)
        X2, y2, t2 = creator2.transform(raw_set)
        self.assertEqual(X1.shape, X2.shape)
        self.assertEqual(len(y1), len(y2))

    # --- Transform on different data after fit ---

    def test_transform_different_data_same_signal_length(self):
        """Transform on new data with same signal length should work."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()],
            window_length=0.5, overlap=0.5,
        )
        raw_set1 = self.generate_sample_data(
            signal_number=5, column_number=3, samples_number=20,
        )
        raw_set2 = self.generate_sample_data(
            signal_number=3, column_number=3, samples_number=20,
        )
        creator.fit(raw_set1)
        X, y, t = creator.transform(raw_set2)
        self.assertEqual(X.shape[0], 3)
        self.assertEqual(X.shape[1], 9)  # 1 * 3 windows * 3 channels

    # --- Constant signal tests ---

    def test_constant_signal_produces_zero_ssc(self):
        """SSC on a constant signal should produce zero features."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorSsc()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=2, column_number=2, samples_number=20,
            class_indices=[0],
        )
        # Replace with constant signals
        for sig in raw_set:
            sig.signal = np.ones_like(sig.signal) * 5.0
        X, y, t = creator.fit_transform(raw_set)
        # SSC counts slope sign changes; constant signal → 0
        np.testing.assert_array_equal(X, np.zeros_like(X))

    def test_constant_signal_mav(self):
        """MAV on a constant signal should produce the constant value."""
        creator = SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()],
            window_length=0.5, overlap=0.5,
        )
        raw_set = self.generate_sample_data(
            signal_number=1, column_number=1, samples_number=20,
            class_indices=[0],
        )
        for sig in raw_set:
            sig.signal = np.ones_like(sig.signal) * 3.0
        X, y, t = creator.fit_transform(raw_set)
        # MAV of constant 3.0 should be 3.0 for each window
        np.testing.assert_allclose(X, 3.0)
