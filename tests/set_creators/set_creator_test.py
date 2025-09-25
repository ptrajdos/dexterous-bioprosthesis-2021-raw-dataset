import unittest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import (
    SetCreator,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_transformer_wrapper import (
    SetCreatorTransformerWrapper,
)
from tests.testing_tools import generate_sample_data
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class SetCreatorTest(unittest.TestCase):

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise unittest.SkipTest("Skipping")

    def get_creators(self) -> SetCreator:
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

        self.assertTrue(X.shape[0] == n_samples, "X -- wrong number of objects")
        self.assertTrue(len(y) == n_samples, "y -- wrong number of objects")
        self.assertTrue(len(t) == n_samples, "t -- wrong number of objects")

    def test_creator_fit_transform(self):

        creators = self.get_creators()
        for signal_number, column_number, samples_number, class_indices in self.get_sample_data_parameters():
            for creator in creators:
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
            with self.subTest(dtype=dtype):
                for creator in creators:

                    raw_set = self.generate_sample_data(samples_number=self.get_default_sample_number(), dtype=dtype)
                    n_samples = len(raw_set)

                    creator.fit(raw_set)
                    X, y, t = creator.transform(raw_set)

                    self.assertTrue(
                        np.issubdtype(X.dtype, dtype),
                        "Wrong dtype of the X. Expected {}, got {}".format(
                            dtype, X.dtype
                        ),
                    )

    def test_creator_fit_then_transform(self):

        creators = self.get_creators()

        for creator in creators:

            raw_set = self.generate_sample_data(samples_number=self.get_default_sample_number())
            n_samples = len(raw_set)

            creator.fit(raw_set)
            X, y, t = creator.transform(raw_set)

            self.basic_test_check(raw_set, X, y, t)

    def test_attributes_indices(self):
        creators = self.get_creators()

        for creator in creators:

            raw_set = self.generate_sample_data(samples_number=self.get_default_sample_number())
            n_samples = len(raw_set)
            n_channels = raw_set.signal_n_cols

            creator.fit(raw_set)
            X, y, t = creator.transform(raw_set)

            n_attribs = X.shape[1]

            indices = creator.get_channel_attribs_indices()

            if indices is not None:
                self.assertTrue(len(indices) == n_channels, "Wrong number of channels")

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

    def test_pipeline(self):
        creators = self.get_creators()

        for creator in creators:

            raw_set = self.generate_sample_data(samples_number=self.get_default_sample_number())

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

        for creator in creators:

            raw_set = self.generate_sample_data(samples_number=self.get_default_sample_number())

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

        for creator in creators:

            raw_set = self.generate_sample_data(samples_number=self.get_default_sample_number())
            n_samples = len(raw_set)
            try:
                X, y, t = creator.transform(raw_set)
                self.fail("Applying transform on unfitted model!")
            except NotFittedError as ex:
                pass
            except Exception as ex:
                self.fail("An exception has been caught: {}".format(ex))


if __name__ == "__main__":
    unittest.main()
