from typing import Dict
import unittest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_column_name_regex import (
    RawSignalsFilterColumnNameRegex,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_pipeline import (
    SetCreatorPipeline,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import (
    SetCreator,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_transformer_wrapper import (
    SetCreatorTransformerWrapper,
)
from tests.testing_tools import (
    generate_one_data,
    generate_sample_data,
    generate_zero_data,
)
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class SetCreatorTest(unittest.TestCase):

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise unittest.SkipTest("Skipping")

    def get_creators(self) -> Dict[str, SetCreator]:
        raise unittest.SkipTest("Skipping")

    def generate_sample_data(
        self,
        signal_number=10,
        column_number=3,
        samples_number=12,
        class_indices=[0, 1],
        dtype=np.double,
    ):
        return generate_sample_data(
            signal_number=signal_number,
            column_number=column_number,
            samples_number=samples_number,
            class_indices=class_indices,
            dtype=dtype,
        )

    def generate_z(
        self,
        signal_number=10,
        column_number=3,
        samples_number=12,
        class_indices=[0, 1],
        dtype=np.double,
    ):
        return generate_zero_data(
            signal_number=signal_number,
            column_number=column_number,
            samples_number=samples_number,
            class_indices=class_indices,
            dtype=dtype,
        )

    def generate_one(
        self,
        signal_number=10,
        column_number=3,
        samples_number=12,
        class_indices=[0, 1],
        dtype=np.double,
    ):
        return generate_one_data(
            signal_number=signal_number,
            column_number=column_number,
            samples_number=samples_number,
            class_indices=class_indices,
            dtype=dtype,
        )

    def get_sample_data_parameters(self):
        return [
            (10, 3, 10, [0, 1]),
            (10, 1, 11, [0, 1]),
        ]

    def get_default_sample_number(self):
        return 12

    def basic_test_check(self, raw_set, X, y, t):

        n_samples = len(raw_set)

        self.assertIsNotNone(X, "Dataset, object description is None")
        self.assertIsNotNone(y, "Class set is None")
        self.assertIsNotNone(t, "Timestamps is none")

        self.assertTrue(
            X.shape[0] == n_samples,
            f"X -- wrong number of objects. Expected {n_samples}, got {X.shape[0]}",
        )
        self.assertTrue(
            len(y) == n_samples,
            f"y -- wrong number of objects. Expected {n_samples}, got {len(y)}",
        )
        self.assertTrue(
            len(t) == n_samples,
            f"t -- wrong number of objects. Expected {n_samples}, got {len(t)}",
        )

        self.assertFalse(np.isnan(X).any(), "Nans in X")
        self.assertFalse(np.isinf(X).any(), "Infs in X")
        self.assertFalse(np.isnan(y).any(), "Nans in y")
        self.assertFalse(np.isinf(y).any(), "Infs in y")
        self.assertFalse(np.isnan(t).any(), "Nans in t")
        self.assertFalse(np.isinf(t).any(), "Infs in t")

    def test_creator_fit_transform(self):

        creators = self.get_creators()
        for (
            signal_number,
            column_number,
            samples_number,
            class_indices,
        ) in self.get_sample_data_parameters():
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

    def test_dtype(self):
        creators = self.get_creators()

        dtypes = [np.float32, np.float64, np.single, np.double]
        for dtype in dtypes:

            for creator_name, creator in creators.items():
                with self.subTest(dtype=dtype, creator_name=creator_name):

                    raw_set = self.generate_sample_data(
                        samples_number=self.get_default_sample_number(), dtype=dtype
                    )
                    n_samples = len(raw_set)

                    creator.fit(raw_set)
                    X, y, t = creator.transform(raw_set)

                    self.assertTrue(
                        np.issubdtype(X.dtype, dtype),
                        "Wrong dtype of the X. Expected {}, got {}".format(
                            dtype, X.dtype
                        ),
                    )

    def test_ydtype(self):
        creators = self.get_creators()

        dtypes = [np.int64, np.int32, np.float32, np.str_]
        classes = np.asanyarray([1, 2, 3])
        for dtype in dtypes:
            for creator_name, creator in creators.items():
                with self.subTest(dtype=dtype, creator_name=creator_name):

                    raw_set = self.generate_sample_data(
                        samples_number=self.get_default_sample_number(),
                        class_indices=classes.astype(dtype),
                    )
                    n_samples = len(raw_set)

                    creator.fit(raw_set)
                    X, y, t = creator.transform(raw_set)
                    labels = y
                    self.assertIsNotNone(labels, "Labels are none!")
                    self.assertIsInstance(
                        labels, np.ndarray, "Wrong labels array type."
                    )
                    self.assertTrue(np.can_cast(labels.dtype, dtype), "Cannot cast")

                    if dtype != np.str_:
                        self.assertTrue(
                            labels.dtype == dtype,
                            f"Wrong exact type. Got {labels.dtype} expect: {dtype} ",
                        )

    def test_creator_fit_then_transform(self):

        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )
                n_samples = len(raw_set)

                creator.fit(raw_set)
                X, y, t = creator.transform(raw_set)

                self.basic_test_check(raw_set, X, y, t)

    def test_creator_zeros(self):

        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_z(
                    samples_number=self.get_default_sample_number()
                )
                n_samples = len(raw_set)

                creator.fit(raw_set)
                X, y, t = creator.transform(raw_set)

                self.basic_test_check(raw_set, X, y, t)

    def test_creator_ones(self):

        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_one(
                    samples_number=self.get_default_sample_number()
                )
                n_samples = len(raw_set)

            creator.fit(raw_set)
            X, y, t = creator.transform(raw_set)

            self.basic_test_check(raw_set, X, y, t)

    def check_attributes_indices(self, creator, X, n_channels):
        """Check that channel attribute indices are valid, non-overlapping, and cover all features.

        Can be reused in future tests that need to validate channel attribute indices.

        Arguments:
        ---------
        creator -- a fitted SetCreator instance
        X -- the feature matrix produced by the creator
        n_channels -- expected number of channels
        """
        n_attribs = X.shape[1]
        indices = creator.get_channel_attribs_indices()

        if indices is not None:
            self.assertTrue(
                len(indices) == n_channels,
                f"Wrong number of channels. Expected {n_channels}, got {len(indices)}",
            )

            for channel_indices in indices:
                try:
                    Xs = X[:, channel_indices]
                except:
                    self.fail("Wrong channel indices")

            indices_coverage = [0 for _ in range(n_attribs)]

            for channel_indices in indices:
                for index in channel_indices:
                    indices_coverage[index] += 1

            self.assertTrue(
                all([idx_count == 1 for idx_count in indices_coverage]),
                "Wrong coverage",
            )

    def test_channel_names(self):
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )
                expected_names = raw_set.get_channel_names()

                creator.fit(raw_set)
                names = creator.get_channel_names()

                self.assertEqual(
                    names, expected_names,
                    f"Channel names mismatch. Expected {expected_names}, got {names}",
                )

    def test_attributes_indices(self):
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )
                n_channels = raw_set.signal_n_cols

                creator.fit(raw_set)
                X, y, t = creator.transform(raw_set)

                self.check_attributes_indices(creator, X, n_channels)

    def test_pipeline(self):
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )

                pipeline = Pipeline(
                    [
                        ("trans", SetCreatorTransformerWrapper(creator)),
                        ("classifier", DecisionTreeClassifier()),
                    ]
                )

                y = raw_set.get_labels()
                pipeline.fit(raw_set, y)

                y_pred = pipeline.predict(raw_set)

    def test_gridsearch(self):
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):

                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )

                pipeline = Pipeline(
                    [
                        ("trans", SetCreatorTransformerWrapper(creator)),
                        ("classifier", DecisionTreeClassifier()),
                    ]
                )
                params = [{"classifier__criterion": ["gini", "entropy"]}]

                y = raw_set.get_labels()
                gs = GridSearchCV(pipeline, param_grid=params, scoring="accuracy", cv=3)

                gs.fit(raw_set, y)
                y_pred = gs.predict(raw_set)

                self.assertIsNotNone(y_pred, "Predictions are none")
                self.assertTrue(len(y) == len(y_pred), "Wrong predictions length")

    def test_not_fitted(self):

        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )
                n_samples = len(raw_set)
                try:
                    X, y, t = creator.transform(raw_set)
                    self.fail("Applying transform on unfitted model!")
                except NotFittedError as ex:
                    pass
                except Exception as ex:
                    self.fail("An exception has been caught: {}".format(ex))

    def test_pipeline_with_rs_filters(self):
        creators = self.get_creators()

        raw_set = self.generate_sample_data(
            samples_number=self.get_default_sample_number(),
            column_number=10,
        )

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                pipeline = SetCreatorPipeline(
                    [
                        (
                            "filter",
                            RawSignalsFilterColumnNameRegex(pattern="^C[0-5]"),
                        ),
                        ("creator", creator),
                    ]
                )
                pipeline.fit(raw_set)
                X, y, t = pipeline.transform(raw_set)

                self.basic_test_check(raw_set, X, y, t)
                n_remaining_channels = 6
                self.check_attributes_indices(pipeline, X, n_remaining_channels)

    def test_pipeline_with_rs_filters_no_selection(self):
        creators = self.get_creators()

        raw_set = self.generate_sample_data(
            samples_number=self.get_default_sample_number(),
            column_number=10,
        )

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                pipeline = SetCreatorPipeline(
                    [
                        (
                            "filter",
                            RawSignalsFilterColumnNameRegex(pattern="^X[0-5]"),
                        ),
                        ("creator", creator),
                    ]
                )
                pipeline.fit(raw_set)
                X, y, t = pipeline.transform(raw_set)

                self.basic_test_check(raw_set, X, y, t)
                n_remaining_channels = 0 # No channels selected
                self.check_attributes_indices(pipeline, X, n_remaining_channels)

    def test_feature_names_out(self):
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )

                creator.fit(raw_set)
                X, y, t = creator.transform(raw_set)
                names = creator.get_feature_names_out()

                self.assertIsInstance(names, np.ndarray, "Wrong type of names")
                self.assertTrue(
                    len(names) == X.shape[1],
                    f"Wrong number of names. Expected {X.shape[1]}, got {len(names)}",
                )
                self.assertTrue(
                    all(isinstance(n, str) for n in names), "Names should be strings"
                )
                self.assertTrue(len(set(names)) == len(names), "Names are not unique")

    def test_feature_names_out_channel_attribs(self):
        """Every attribute assigned to a channel should be named after that channel."""
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )

                creator.fit(raw_set)
                names = creator.get_feature_names_out()
                indices = creator.get_channel_attribs_indices()
                if indices is None:
                    continue

                for channel_name, channel_indices in zip(
                    creator.get_channel_names(), indices
                ):
                    for idx in channel_indices:
                        self.assertTrue(
                            names[idx].startswith(channel_name + "_"),
                            f"Attribute {names[idx]} not named after channel {channel_name}",
                        )

    def test_feature_names_out_not_fitted(self):
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                with self.assertRaises(NotFittedError):
                    creator.get_feature_names_out()

    def test_pandas_output(self):
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )

                returned = creator.set_output(transform="pandas")
                self.assertIs(returned, creator, "set_output should return self")

                X_ft, y_ft, t_ft = creator.fit_transform(raw_set)
                X, y, t = creator.transform(raw_set)

                for X_df in (X_ft, X):
                    self.assertIsInstance(X_df, pd.DataFrame, "X is not a DataFrame")
                    self.assertTrue(
                        list(X_df.columns) == list(creator.get_feature_names_out()),
                        "Wrong column names",
                    )
                self.basic_test_check(raw_set, X.to_numpy(), y, t)

                creator.set_output(transform="default")
                X_np, _, _ = creator.transform(raw_set)
                self.assertIsInstance(X_np, np.ndarray, "Default output is not numpy")
                self.assertTrue(X_np.shape == X.shape, "Shapes differ")

    def test_pandas_output_wrong_value(self):
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                with self.assertRaises(ValueError):
                    creator.set_output(transform="polars_or_whatever")

    def test_pandas_output_pipeline(self):
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )

                pipeline = Pipeline(
                    [
                        ("trans", SetCreatorTransformerWrapper(creator)),
                        ("classifier", DecisionTreeClassifier()),
                    ]
                ).set_output(transform="pandas")

                y = raw_set.get_labels()
                pipeline.fit(raw_set, y)

                self.assertTrue(
                    list(pipeline["classifier"].feature_names_in_)
                    == list(pipeline["trans"].get_feature_names_out()),
                    "Classifier did not receive attribute names",
                )
                pipeline.predict(raw_set)

    def test_pandas_output_set_creator_pipeline(self):
        creators = self.get_creators()

        raw_set = self.generate_sample_data(
            samples_number=self.get_default_sample_number(),
            column_number=10,
        )

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                pipeline = SetCreatorPipeline(
                    [
                        (
                            "filter",
                            RawSignalsFilterColumnNameRegex(pattern="^C[0-5]"),
                        ),
                        ("creator", creator),
                    ]
                ).set_output(transform="pandas")
                pipeline.fit(raw_set)
                X, y, t = pipeline.transform(raw_set)

                self.assertIsInstance(X, pd.DataFrame, "X is not a DataFrame")
                self.assertTrue(
                    list(X.columns) == list(pipeline.get_feature_names_out()),
                    "Wrong column names",
                )

    def test_sklearn_is_fitted(self):
        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                raw_set = self.generate_sample_data(
                    samples_number=self.get_default_sample_number()
                )
                wrapper = SetCreatorTransformerWrapper(creator)

                self.assertFalse(creator.__sklearn_is_fitted__(), "Should not be fitted")
                self.assertFalse(wrapper.__sklearn_is_fitted__(), "Wrapper should not be fitted")

                wrapper.fit(raw_set)

                self.assertTrue(creator.__sklearn_is_fitted__(), "Should be fitted")
                self.assertTrue(wrapper.__sklearn_is_fitted__(), "Wrapper should be fitted")

    def test_sklearn_tags(self):
        try:
            from sklearn.utils import get_tags
        except ImportError:
            self.skipTest("Estimator tags API requires scikit-learn >= 1.6")

        creators = self.get_creators()

        for creator_name, creator in creators.items():
            with self.subTest(creator_name=creator_name):
                for obj in (creator, SetCreatorTransformerWrapper(creator)):
                    tags = get_tags(obj)
                    self.assertTrue(tags.requires_fit, "Set creator requires fit")
                    self.assertIsNotNone(tags.transformer_tags, "Should be a transformer")


if __name__ == "__main__":
    unittest.main()
