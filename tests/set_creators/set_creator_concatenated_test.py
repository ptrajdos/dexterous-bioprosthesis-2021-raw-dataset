import unittest

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mav import (
    NpSignalExtractorMav,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ssc import (
    NpSignalExtractorSsc,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ar import (
    NpSignalExtractorAr,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_functions import (
    SetCreatorFunctions,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_concatenated import (
    SetCreatorConcatenated,
)
from tests.set_creators.set_creator_test import SetCreatorTest
from tests.testing_tools import generate_sample_data


class SetCreatorConcatenatedTest(SetCreatorTest):

    __test__ = True

    def get_creators(self):
        return {
            "concat_two_creators": SetCreatorConcatenated(
                creators=[
                    SetCreatorFunctions(extractors=[NpSignalExtractorMav()]),
                    SetCreatorFunctions(extractors=[NpSignalExtractorSsc()]),
                ]
            ),
        }

    def test_channel_names(self):
        """Override: concatenated creator deduplicates channel names from sub-creators."""
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )
                expected_names = raw_set.get_channel_names()

                creator.fit(raw_set)
                names = creator.get_channel_names()

                # Same channels used by both sub-creators, so deduplicated names should match raw
                self.assertEqual(
                    names,
                    expected_names,
                    f"Channel names mismatch. Expected {expected_names}, got {names}",
                )

    def test_concatenated_feature_count(self):
        """Total features should equal sum of features from all sub-creators."""
        c1 = SetCreatorFunctions(extractors=[NpSignalExtractorMav()])
        c2 = SetCreatorFunctions(extractors=[NpSignalExtractorSsc()])
        cc = SetCreatorConcatenated(creators=[c1, c2])

        raw_set = self.generate_sample_data(
            samples_number=self.get_default_sample_number()
        )
        cc.fit(raw_set)
        X_cc, _, _ = cc.transform(raw_set)

        # Fit individual creators separately to get their feature counts
        c1_solo = SetCreatorFunctions(extractors=[NpSignalExtractorMav()])
        c2_solo = SetCreatorFunctions(extractors=[NpSignalExtractorSsc()])
        c1_solo.fit(raw_set)
        c2_solo.fit(raw_set)
        X1, _, _ = c1_solo.transform(raw_set)
        X2, _, _ = c2_solo.transform(raw_set)

        self.assertEqual(
            X_cc.shape[1],
            X1.shape[1] + X2.shape[1],
            f"Feature count mismatch: {X_cc.shape[1]} != {X1.shape[1]} + {X2.shape[1]}",
        )

    def test_concatenated_attrib_indices_coverage(self):
        """Attribute indices should cover all features without overlap."""
        c1 = SetCreatorFunctions(extractors=[NpSignalExtractorMav()])
        c2 = SetCreatorFunctions(extractors=[NpSignalExtractorSsc()])
        cc = SetCreatorConcatenated(creators=[c1, c2])

        raw_set = self.generate_sample_data(
            samples_number=self.get_default_sample_number()
        )
        cc.fit(raw_set)
        X, _, _ = cc.transform(raw_set)

        n_channels = len(cc.get_channel_names())
        self.check_attributes_indices(cc, X, n_channels)

    def test_concatenated_three_creators(self):
        """Concatenation should work with three sub-creators."""
        cc = SetCreatorConcatenated(
            creators=[
                SetCreatorFunctions(extractors=[NpSignalExtractorMav()]),
                SetCreatorFunctions(extractors=[NpSignalExtractorSsc()]),
                SetCreatorFunctions(extractors=[NpSignalExtractorAr()]),
            ]
        )

        raw_set = self.generate_sample_data(
            samples_number=self.get_default_sample_number()
        )
        cc.fit(raw_set)
        X, y, t = cc.transform(raw_set)

        self.basic_test_check(raw_set, X, y, t)
        n_channels = len(cc.get_channel_names())
        self.check_attributes_indices(cc, X, n_channels)

    def test_concatenated_single_creator(self):
        """Concatenation with a single sub-creator should match that creator's output."""
        c1 = SetCreatorFunctions(extractors=[NpSignalExtractorMav()])
        cc = SetCreatorConcatenated(creators=[c1])

        raw_set = self.generate_sample_data(
            samples_number=self.get_default_sample_number()
        )

        c1_solo = SetCreatorFunctions(extractors=[NpSignalExtractorMav()])
        c1_solo.fit(raw_set)
        X1, y1, t1 = c1_solo.transform(raw_set)

        cc.fit(raw_set)
        X_cc, y_cc, t_cc = cc.transform(raw_set)

        np.testing.assert_array_equal(X_cc, X1)
        np.testing.assert_array_equal(y_cc, y1)
        np.testing.assert_array_equal(t_cc, t1)

    def test_concatenated_labels_preserved(self):
        """Labels and timestamps should come from the first sub-creator."""
        cc = SetCreatorConcatenated(
            creators=[
                SetCreatorFunctions(extractors=[NpSignalExtractorMav()]),
                SetCreatorFunctions(extractors=[NpSignalExtractorSsc()]),
            ]
        )

        raw_set = self.generate_sample_data(
            samples_number=self.get_default_sample_number()
        )
        cc.fit(raw_set)
        X, y, t = cc.transform(raw_set)

        expected_labels = raw_set.get_labels()
        np.testing.assert_array_equal(y, expected_labels)

    def test_concatenated_same_extractors(self):
        """Two creators with the same extractor should double the feature count."""
        c1 = SetCreatorFunctions(extractors=[NpSignalExtractorMav()])
        c2 = SetCreatorFunctions(extractors=[NpSignalExtractorMav()])
        cc = SetCreatorConcatenated(creators=[c1, c2])

        raw_set = self.generate_sample_data(
            samples_number=self.get_default_sample_number()
        )
        cc.fit(raw_set)
        X_cc, _, _ = cc.transform(raw_set)

        c_solo = SetCreatorFunctions(extractors=[NpSignalExtractorMav()])
        c_solo.fit(raw_set)
        X_solo, _, _ = c_solo.transform(raw_set)

        self.assertEqual(X_cc.shape[1], 2 * X_solo.shape[1])

    def test_concatenated_channel_names_deduplicated(self):
        """When sub-creators share channels, names should be deduplicated."""
        c1 = SetCreatorFunctions(extractors=[NpSignalExtractorMav()])
        c2 = SetCreatorFunctions(extractors=[NpSignalExtractorSsc()])
        cc = SetCreatorConcatenated(creators=[c1, c2])

        raw_set = self.generate_sample_data(
            samples_number=self.get_default_sample_number(), column_number=3
        )
        cc.fit(raw_set)

        names = cc.get_channel_names()
        # Both sub-creators use the same raw signals, so names should equal raw channel names
        self.assertEqual(len(names), 3)
        # No duplicates
        self.assertEqual(len(names), len(set(names)))

    def test_concatenated_fit_transform(self):
        """fit_transform should produce the same result as fit then transform."""
        raw_set = self.generate_sample_data(
            samples_number=self.get_default_sample_number()
        )

        cc1 = SetCreatorConcatenated(
            creators=[
                SetCreatorFunctions(extractors=[NpSignalExtractorMav()]),
                SetCreatorFunctions(extractors=[NpSignalExtractorSsc()]),
            ]
        )
        X1, y1, t1 = cc1.fit_transform(raw_set)

        cc2 = SetCreatorConcatenated(
            creators=[
                SetCreatorFunctions(extractors=[NpSignalExtractorMav()]),
                SetCreatorFunctions(extractors=[NpSignalExtractorSsc()]),
            ]
        )
        cc2.fit(raw_set)
        X2, y2, t2 = cc2.transform(raw_set)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(t1, t2)


if __name__ == "__main__":
    unittest.main()
