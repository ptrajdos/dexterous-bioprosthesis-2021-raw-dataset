"""Regression tests for set creator bugs (fs propagation, dtypes, side effects)."""
import unittest

import numpy as np
from sklearn.base import clone

from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator_dtw_fast import (
    DistanceMatrixCalculatorDTWFast,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mav import (
    NpSignalExtractorMav,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_cluster_dist import (
    SetCreatorClusterDist,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_concatenated import (
    SetCreatorConcatenated,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_dwt import SetCreatorDWT
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_functions import (
    SetCreatorFunctions,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_mds import (
    SetCreatorMDS,
    default_ga_options,
    euclid_flatten,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_swt import SetCreatorSWT
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_transformer_wrapper import (
    SetCreatorTransformerWrapper,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_window_delta import (
    SetCreatorWindowDelta,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_window_delta_diff import (
    SetCreatorWindowDeltaDiff,
)


class FsSpy(NPSignalExtractor):
    """Returns the sample rate it was fitted with."""

    def _transform(self, X):
        return np.full(X.shape[1], float(self._get_fs()))

    def attribs_per_column(self):
        return 1


def make_signals(n=4, length=64, fs=500, n_channels=2, dtype=float):
    rng = np.random.default_rng(0)
    signals = RawSignals(sample_rate=fs)
    for i in range(n):
        data = (rng.normal(size=(length, n_channels)) * 100).astype(dtype)
        signals.append(RawSignal(data, object_class=i % 2))
    return signals


FAST_GA = dict(default_ga_options, num_generations=5)


class SetCreatorsRegressionTest(unittest.TestCase):

    def test_fs_passed_to_extractors(self):
        for creator in [
            SetCreatorFunctions([FsSpy()]),
            SetCreatorWindowDelta([FsSpy()]),
        ]:
            with self.subTest(creator=type(creator).__name__):
                X, _, _ = creator.fit_transform(make_signals(fs=500))
                self.assertTrue(np.all(X == 500))

    def test_dwt_level_sample_rates(self):
        creator = SetCreatorDWT(num_levels=2, extractors=[FsSpy()])
        X, _, _ = creator.fit_transform(make_signals(fs=500, n_channels=1))
        # [A2, D2, D1]
        self.assertTrue(np.array_equal(X[0], [125, 125, 250]))

    def test_integer_signals_give_float_features(self):
        X, _, _ = SetCreatorFunctions([NpSignalExtractorMav()]).fit_transform(
            make_signals(dtype=np.int16)
        )
        self.assertTrue(np.issubdtype(X.dtype, np.floating))
        self.assertFalse(np.all(X == np.round(X)))

    def test_window_delta_requires_equal_lengths(self):
        for cls in [SetCreatorWindowDelta, SetCreatorWindowDeltaDiff]:
            with self.subTest(creator=cls.__name__):
                signals = make_signals(n=1, length=100)
                signals.append(RawSignal(np.ones((101, 2))))
                with self.assertRaises(ValueError):
                    cls([NpSignalExtractorMav()]).fit(signals)

                creator = cls([NpSignalExtractorMav()]).fit(make_signals(length=100))
                with self.assertRaises(ValueError):
                    creator.transform(make_signals(length=120))

    def test_swt_too_short_signal(self):
        creator = SetCreatorSWT(num_levels=4, extractors=[NpSignalExtractorMav()])
        with self.assertRaises(ValueError):
            creator.fit_transform(make_signals(length=10))

    def test_cluster_dist_keeps_sample_rate(self):
        signals = make_signals(n=8, length=30, fs=500)
        SetCreatorClusterDist(DistanceMatrixCalculatorDTWFast()).fit(signals)
        self.assertEqual({s.get_sample_rate() for s in signals}, {500})

    def test_cluster_dist_few_samples(self):
        for n in [1, 2, 3]:
            with self.subTest(n=n):
                X, _, _ = SetCreatorClusterDist(
                    DistanceMatrixCalculatorDTWFast()
                ).fit_transform(make_signals(n=n, length=30))
                self.assertEqual(X.shape[0], n)

    def test_mds_does_not_modify_options(self):
        ga_options = dict(FAST_GA, sol_per_pop_mult=3)
        mds_options = {"n_components": 5, "max_iter": 50}
        creator = SetCreatorMDS(ga_options=ga_options, mds_options=mds_options)
        creator.fit(make_signals(n=6, length=30))
        creator.transform(make_signals(n=2, length=30))
        self.assertEqual(ga_options["sol_per_pop_mult"], 3)
        self.assertEqual(creator.ga_options["sol_per_pop_mult"], 3)
        self.assertIn("n_components", mds_options)
        self.assertIn("sol_per_pop_mult", default_ga_options)

    def test_concatenated_with_flattened_creator(self):
        creator = SetCreatorConcatenated(
            [
                SetCreatorFunctions([NpSignalExtractorMav()]),
                SetCreatorMDS(flatten_function=euclid_flatten, ga_options=FAST_GA),
            ]
        )
        X, _, _ = creator.fit_transform(make_signals(n=6, length=30))
        self.assertEqual(X.shape, (6, 4))
        X_new, _, _ = creator.transform(make_signals(n=2, length=30))
        self.assertEqual(X_new.shape, (2, 4))
        self.assertTrue(creator.__sklearn_is_fitted__())

    def test_wrapper_clone_and_defaults(self):
        wrapper = SetCreatorTransformerWrapper(SetCreatorFunctions([NpSignalExtractorMav()]))
        cloned = clone(wrapper)
        self.assertIsNot(cloned.set_creator, wrapper.set_creator)

        a, b = SetCreatorTransformerWrapper(), SetCreatorTransformerWrapper()
        self.assertIsNot(a._get_set_creator(), b._get_set_creator())


if __name__ == "__main__":
    unittest.main()
