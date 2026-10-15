import unittest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from dexterous_bioprosthesis_2021_raw_datasets.preprocessing.uniform_transformer import UniformTransformer

class UniformTransformerTest(unittest.TestCase):
    

    def get_transformer(self):
        return UniformTransformer()
    
    def test_transforms(self):
        transfomer = self.get_transformer()

        X, y = load_iris(return_X_y=True)

        transfomer.fit(X,y)
        
        Xt  = transfomer.transform(X)
        self.assertIsNotNone(Xt, "Transformed set is none")
        self.assertIsInstance(Xt,type(X), "Wrong output type")
        
        self.assertTrue(Xt.shape[1] == X.shape[1], "Wrong number of columns!")

    def test_fittransforms(self):
        transfomer = self.get_transformer()

        X, y = load_iris(return_X_y=True)
        
        Xt  = transfomer.fit_transform(X)
        self.assertIsNotNone(Xt, "Transformed set is none")
        self.assertIsInstance(Xt,type(X), "Wrong output type")
        self.assertTrue(Xt.shape[1] == X.shape[1], "Wrong number of columns!")

    def test_in_pipeline(self):
        transformer = self.get_transformer()

        X,y =  load_iris(return_X_y=True)

        pipeline = Pipeline(steps=[('transformer',transformer), ('classifier', RandomForestClassifier(n_estimators=4))])

        pipeline.fit(X,y)
        y_pred = pipeline.predict(X)

        self.assertIsNotNone(y_pred, "Predictions none")
        self.assertIsInstance(y_pred, type(y), "Wrong predictions type")

    def test_pandas(self):
        transformer = self.get_transformer()

        X,y =  load_iris(return_X_y=True)

        pipeline = Pipeline(steps=[('transformer',transformer), ('classifier', RandomForestClassifier(n_estimators=4))])

        Xp  = pd.DataFrame(X)

        pipeline.fit(Xp,y)
        y_pred = pipeline.predict(Xp)

        self.assertIsNotNone(y_pred, "Predictions none")
        self.assertIsInstance(y_pred, type(y), "Wrong predictions type")

    def test_element_fraction(self):
        X, y = load_iris(return_X_y=True)
        n_rows, n_cols = X.shape

        for element_fraction, expected_rows in [
            (1.0, n_rows),
            (0.5, n_rows // 2),
            (2.0, 2 * n_rows),
            (1e-6, 1),  # at least one row
        ]:
            with self.subTest(element_fraction=element_fraction):
                Xt = UniformTransformer(element_fraction=element_fraction).fit_transform(X)
                self.assertTrue(
                    Xt.shape == (expected_rows, n_cols),
                    f"Wrong shape. Expected {(expected_rows, n_cols)}, got {Xt.shape}",
                )

    def test_value_range(self):
        np.random.seed(0)
        X = np.random.uniform(low=-2.0, high=3.0, size=(50, 4))
        mins, maxs = X.min(axis=0), X.max(axis=0)

        for extend_factor in [0.0, 0.1, 0.5]:
            with self.subTest(extend_factor=extend_factor):
                Xt = UniformTransformer(
                    extend_factor=extend_factor, element_fraction=200
                ).fit_transform(X)

                low = mins - extend_factor * np.abs(mins)
                high = maxs + extend_factor * np.abs(maxs)
                self.assertTrue(np.all(Xt >= low), "Values below the extended range")
                self.assertTrue(np.all(Xt <= high), "Values above the extended range")

                if extend_factor > 0:
                    # 10000 samples per column: the extended parts of the range are hit
                    self.assertTrue(np.all(Xt.min(axis=0) < mins), "Lower bound not extended")
                    self.assertTrue(np.all(Xt.max(axis=0) > maxs), "Upper bound not extended")

    def test_pandas_output(self):
        X, y = load_iris(return_X_y=True, as_frame=True)
        n_rows, n_cols = X.shape

        for element_fraction, expected_rows in [(1.0, n_rows), (0.5, n_rows // 2), (2.0, 2 * n_rows)]:
            with self.subTest(element_fraction=element_fraction):
                np.random.seed(0)
                Xt = UniformTransformer(element_fraction=element_fraction).fit_transform(X)
                np.random.seed(0)
                Xt_np = UniformTransformer(element_fraction=element_fraction).fit_transform(
                    X.to_numpy()
                )

                self.assertIsInstance(Xt, pd.DataFrame, "Wrong output type")
                self.assertTrue(list(Xt.columns) == list(X.columns), "Wrong column names")
                self.assertTrue(Xt.shape == (expected_rows, n_cols), "Wrong shape")
                self.assertTrue(
                    Xt.index.equals(pd.RangeIndex(expected_rows)), "Wrong index"
                )
                self.assertTrue(np.allclose(Xt.to_numpy(), Xt_np), "Different values")

    def test_pandas_non_default_index(self):
        """Input index is not reused, the rows are generated."""
        X, y = load_iris(return_X_y=True, as_frame=True)
        X.index = X.index + 1000

        Xt = UniformTransformer().fit_transform(X)
        self.assertIsInstance(Xt, pd.DataFrame, "Wrong output type")
        self.assertTrue(Xt.index.equals(pd.RangeIndex(len(X))), "Wrong index")
