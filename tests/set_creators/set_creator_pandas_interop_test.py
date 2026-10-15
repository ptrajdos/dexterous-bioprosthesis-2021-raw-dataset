"""Interoperation of set creators returning pandas DataFrames with other algorithms.

Each algorithm is run twice: on the default (numpy) output and on the pandas output
of the same set creator. Results should be the same; pandas output should
additionally carry the attribute names.
"""
import random
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dexterous_bioprosthesis_2021_raw_datasets.estimators.estimator_with_extractor import (
    EstimatorWithExtractor,
)
from dexterous_bioprosthesis_2021_raw_datasets.preprocessing.attribute_reorder import (
    AttributeReorder,
)
from dexterous_bioprosthesis_2021_raw_datasets.preprocessing.one_class.outlier_generators.outlier_generator_uniform import (
    OutlierGeneratorUniform,
)
from dexterous_bioprosthesis_2021_raw_datasets.preprocessing.select_attributes_transformer import (
    SelectAttributesTransformer,
)
from dexterous_bioprosthesis_2021_raw_datasets.preprocessing.uniform_transformer import (
    UniformTransformer,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ar import (
    NpSignalExtractorAr,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mav import (
    NpSignalExtractorMav,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_rms import (
    NpSignalExtractorRms,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_concatenated import (
    SetCreatorConcatenated,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_dwt import (
    SetCreatorDWT,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_functions import (
    SetCreatorFunctions,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_window_delta import (
    SetCreatorWindowDelta,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_transformer_wrapper import (
    SetCreatorTransformerWrapper,
)
from tests.testing_tools import generate_sample_data


def get_creator_factories():
    """Factories, so that numpy and pandas runs use separate creator instances."""
    return {
        "functions": lambda: SetCreatorFunctions(
            extractors=[NpSignalExtractorMav(), NpSignalExtractorAr(lags=1)]
        ),
        "dwt": lambda: SetCreatorDWT(
            num_levels=2, extractors=[NpSignalExtractorMav(), NpSignalExtractorRms()]
        ),
        "window_delta": lambda: SetCreatorWindowDelta(
            extractors=[NpSignalExtractorMav()], window_length=0.5, overlap=0.5
        ),
        "concatenated": lambda: SetCreatorConcatenated(
            creators=[
                SetCreatorFunctions(extractors=[NpSignalExtractorMav()]),
                SetCreatorDWT(num_levels=1, extractors=[NpSignalExtractorRms()]),
            ]
        ),
    }


def get_classifiers():
    """Deterministic classifiers, so that numpy and pandas predictions can be compared."""
    return {
        "decision_tree": lambda: DecisionTreeClassifier(random_state=0),
        "random_forest": lambda: RandomForestClassifier(n_estimators=10, random_state=0),
        "knn": lambda: KNeighborsClassifier(n_neighbors=3),
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "svc": lambda: SVC(),
        "gaussian_nb": lambda: GaussianNB(),
    }


def get_transformers():
    """sklearn transformers. Second element: True if the output columns are a subset of input columns."""
    return {
        "standard_scaler": (lambda: StandardScaler(), True),
        "minmax_scaler": (lambda: MinMaxScaler(), True),
        "variance_threshold": (lambda: VarianceThreshold(), True),
        "select_k_best": (lambda: SelectKBest(k=2), True),
        "pca": (lambda: PCA(n_components=2), False),
    }


class SetCreatorPandasInteropTest(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        self.raw_set = generate_sample_data(
            signal_number=30, column_number=3, samples_number=64, class_indices=[0, 1]
        )
        self.y = self.raw_set.get_labels()

    def _creators_pair(self, factory):
        """Return (numpy creator, pandas creator)."""
        return factory(), factory().set_output(transform="pandas")

    def test_set_creator_output(self):
        for creator_name, factory in get_creator_factories().items():
            with self.subTest(creator=creator_name):
                creator_np, creator_pd = self._creators_pair(factory)

                X_np, y_np, t_np = creator_np.fit_transform(self.raw_set)
                X_pd, y_pd, t_pd = creator_pd.fit_transform(self.raw_set)

                self.assertIsInstance(X_np, np.ndarray)
                self.assertIsInstance(X_pd, pd.DataFrame)
                self.assertTrue(
                    list(X_pd.columns) == list(creator_pd.get_feature_names_out()),
                    "Wrong column names",
                )
                self.assertTrue(np.allclose(X_np, X_pd.to_numpy()), "Different values")
                self.assertTrue(np.array_equal(y_np, y_pd), "Different labels")
                self.assertTrue(np.array_equal(t_np, t_pd), "Different timestamps")

    def test_classifiers(self):
        for creator_name, factory in get_creator_factories().items():
            for clf_name, clf_factory in get_classifiers().items():
                with self.subTest(creator=creator_name, classifier=clf_name):
                    creator_np, creator_pd = self._creators_pair(factory)

                    X_np, y, _ = creator_np.fit_transform(self.raw_set)
                    X_pd, _, _ = creator_pd.fit_transform(self.raw_set)

                    clf_np = clf_factory().fit(X_np, y)
                    clf_pd = clf_factory().fit(X_pd, y)

                    self.assertTrue(
                        list(clf_pd.feature_names_in_) == list(X_pd.columns),
                        "Classifier did not receive attribute names",
                    )
                    self.assertTrue(
                        np.array_equal(clf_np.predict(X_np), clf_pd.predict(X_pd)),
                        "Different predictions for numpy and pandas input",
                    )

    def test_classifiers_pipeline(self):
        for creator_name, factory in get_creator_factories().items():
            for clf_name, clf_factory in get_classifiers().items():
                with self.subTest(creator=creator_name, classifier=clf_name):
                    pipelines = [
                        Pipeline(
                            [
                                ("trans", SetCreatorTransformerWrapper(factory())),
                                ("scaler", StandardScaler()),
                                ("clf", clf_factory()),
                            ]
                        )
                        for _ in range(2)
                    ]
                    pipe_np, pipe_pd = pipelines
                    pipe_pd.set_output(transform="pandas")

                    pipe_np.fit(self.raw_set, self.y)
                    pipe_pd.fit(self.raw_set, self.y)

                    self.assertTrue(
                        list(pipe_pd["clf"].feature_names_in_)
                        == list(pipe_pd["trans"].get_feature_names_out()),
                        "Classifier did not receive attribute names",
                    )
                    self.assertTrue(
                        np.array_equal(
                            pipe_np.predict(self.raw_set), pipe_pd.predict(self.raw_set)
                        ),
                        "Different predictions for numpy and pandas pipelines",
                    )

    def test_sklearn_transformers(self):
        for creator_name, factory in get_creator_factories().items():
            for tr_name, (tr_factory, keeps_columns) in get_transformers().items():
                with self.subTest(creator=creator_name, transformer=tr_name):
                    creator_np, creator_pd = self._creators_pair(factory)

                    X_np, y, _ = creator_np.fit_transform(self.raw_set)
                    X_pd, _, _ = creator_pd.fit_transform(self.raw_set)

                    Xt_np = tr_factory().fit_transform(X_np, y)
                    Xt_pd = tr_factory().set_output(transform="pandas").fit_transform(X_pd, y)

                    self.assertIsInstance(Xt_pd, pd.DataFrame, "Output is not a DataFrame")
                    self.assertTrue(
                        np.allclose(Xt_np, Xt_pd.to_numpy()),
                        "Different results for numpy and pandas input",
                    )
                    if keeps_columns:
                        self.assertTrue(
                            set(Xt_pd.columns) <= set(X_pd.columns),
                            "Attribute names have been lost",
                        )

    def test_column_transformer_channel_selection(self):
        """Attributes of a single channel can be selected by name."""
        for creator_name, factory in get_creator_factories().items():
            with self.subTest(creator=creator_name):
                _, creator_pd = self._creators_pair(factory)
                X_pd, y, _ = creator_pd.fit_transform(self.raw_set)

                channel_name = creator_pd.get_channel_names()[0]
                ct = ColumnTransformer(
                    [
                        (
                            "channel",
                            StandardScaler(),
                            make_column_selector(pattern="^{}_".format(channel_name)),
                        )
                    ],
                    verbose_feature_names_out=False,
                ).set_output(transform="pandas")

                Xt = ct.fit_transform(X_pd)

                channel_attribs = creator_pd.get_channel_attribs_indices()[0]
                self.assertTrue(
                    Xt.shape == (len(self.raw_set), len(channel_attribs)),
                    "Wrong number of selected attributes",
                )
                self.assertTrue(
                    list(Xt.columns) == list(X_pd.columns[channel_attribs]),
                    "Selected attributes do not match channel attribute indices",
                )

    def test_model_selection(self):
        for creator_name, factory in get_creator_factories().items():
            with self.subTest(creator=creator_name):
                pipeline = Pipeline(
                    [
                        ("trans", SetCreatorTransformerWrapper(factory())),
                        ("clf", DecisionTreeClassifier(random_state=0)),
                    ]
                ).set_output(transform="pandas")

                cv = KFold(n_splits=3, shuffle=True, random_state=0)
                scores = cross_val_score(pipeline, self.raw_set, self.y, cv=cv)
                self.assertTrue(len(scores) == 3, "Wrong number of scores")
                self.assertFalse(np.isnan(scores).any(), "NaN scores")

                gs = GridSearchCV(
                    pipeline,
                    param_grid={"clf__criterion": ["gini", "entropy"]},
                    cv=cv,
                )
                gs.fit(self.raw_set, self.y)
                y_pred = gs.predict(self.raw_set)
                self.assertTrue(len(y_pred) == len(self.y), "Wrong predictions length")

    def test_estimator_with_extractor(self):
        for creator_name, factory in get_creator_factories().items():
            with self.subTest(creator=creator_name):
                creator_np, creator_pd = self._creators_pair(factory)

                est_np = EstimatorWithExtractor(
                    creator_np, DecisionTreeClassifier(random_state=0)
                ).fit(self.raw_set, self.y)
                est_pd = EstimatorWithExtractor(
                    creator_pd, DecisionTreeClassifier(random_state=0)
                ).fit(self.raw_set, self.y)

                self.assertTrue(
                    np.array_equal(est_np.predict(self.raw_set), est_pd.predict(self.raw_set)),
                    "Different predictions",
                )
                self.assertTrue(
                    np.allclose(
                        est_np.predict_proba(self.raw_set), est_pd.predict_proba(self.raw_set)
                    ),
                    "Different probabilities",
                )

    def test_attribute_reorder(self):
        for creator_name, factory in get_creator_factories().items():
            with self.subTest(creator=creator_name):
                creator_np, creator_pd = self._creators_pair(factory)
                X_np, _, _ = creator_np.fit_transform(self.raw_set)
                X_pd, _, _ = creator_pd.fit_transform(self.raw_set)

                np.random.seed(1)
                Xt_np = AttributeReorder().fit_transform(X_np)
                np.random.seed(1)
                Xt_pd = AttributeReorder().fit_transform(X_pd)

                self.assertIsInstance(Xt_pd, pd.DataFrame)
                self.assertTrue(np.allclose(Xt_np, Xt_pd.to_numpy()), "Different results")
                self.assertTrue(
                    sorted(Xt_pd.columns) == sorted(X_pd.columns),
                    "Attribute names have been lost",
                )
                # Names move with their values
                for column in Xt_pd.columns:
                    self.assertTrue(np.allclose(Xt_pd[column], X_pd[column]))

    def test_select_attributes_transformer(self):
        for creator_name, factory in get_creator_factories().items():
            with self.subTest(creator=creator_name):
                creator_np, creator_pd = self._creators_pair(factory)
                X_np, _, _ = creator_np.fit_transform(self.raw_set)
                X_pd, _, _ = creator_pd.fit_transform(self.raw_set)

                indices = [0, X_np.shape[1] - 1]
                Xt_np = SelectAttributesTransformer(column_indices=indices).fit_transform(X_np)
                Xt_pd = SelectAttributesTransformer(column_indices=indices).fit_transform(X_pd)

                self.assertIsInstance(Xt_pd, pd.DataFrame)
                self.assertTrue(np.allclose(Xt_np, Xt_pd.to_numpy()), "Different results")
                self.assertTrue(
                    list(Xt_pd.columns) == list(X_pd.columns[indices]),
                    "Wrong attribute names",
                )

    def test_uniform_transformer(self):
        for creator_name, factory in get_creator_factories().items():
            with self.subTest(creator=creator_name):
                creator_np, creator_pd = self._creators_pair(factory)
                X_np, _, _ = creator_np.fit_transform(self.raw_set)
                X_pd, _, _ = creator_pd.fit_transform(self.raw_set)

                np.random.seed(1)
                Xt_np = UniformTransformer().fit_transform(X_np)
                np.random.seed(1)
                with warnings.catch_warnings():
                    # e.g. positional Series access, which becomes an error in pandas 3
                    warnings.simplefilter("error", FutureWarning)
                    Xt_pd = UniformTransformer().fit_transform(X_pd)

                self.assertTrue(Xt_np.shape == X_np.shape, "Wrong shape")
                self.assertTrue(
                    np.allclose(Xt_np, np.asanyarray(Xt_pd)), "Different results"
                )

    def test_outlier_generator_uniform(self):
        for creator_name, factory in get_creator_factories().items():
            with self.subTest(creator=creator_name):
                creator_np, creator_pd = self._creators_pair(factory)
                X_np, y, _ = creator_np.fit_transform(self.raw_set)
                X_pd, _, _ = creator_pd.fit_transform(self.raw_set)

                np.random.seed(1)
                out_np, labels_np = OutlierGeneratorUniform().fit(X_np, y).generate()
                np.random.seed(1)
                with warnings.catch_warnings():
                    # e.g. positional Series access, which becomes an error in pandas 3
                    warnings.simplefilter("error", FutureWarning)
                    out_pd, labels_pd = OutlierGeneratorUniform().fit(X_pd, y).generate()

                self.assertTrue(out_np.shape == X_np.shape, "Wrong shape")
                self.assertTrue(
                    np.allclose(out_np, np.asanyarray(out_pd)), "Different results"
                )
                self.assertTrue(np.array_equal(labels_np, labels_pd), "Different labels")


if __name__ == "__main__":
    unittest.main()
