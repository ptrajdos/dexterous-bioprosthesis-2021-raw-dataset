import unittest

from dexterous_bioprosthesis_2021_raw_datasets.preprocessing.select_attributes_transformer import SelectAttributesTransformer
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class SingleAttributeTransformerTest(unittest.TestCase):

    def get_transformer(self):
        return SelectAttributesTransformer()
    
    def test_transforms(self):
        transfomer = self.get_transformer()

        X, y = load_iris(return_X_y=True)

        transfomer.fit(X,y)
        
        Xt  = transfomer.transform(X)
        self.assertIsNotNone(Xt, "Transformed set is none")
        self.assertIsInstance(Xt,type(X), "Wrong output type")
        self.assertTrue(Xt.shape[0] == X.shape[0], "Wrong number of rows")
        self.assertTrue(Xt.shape[1] == 1, "There should be only one colum!")

    def test_fittransforms(self):
        transfomer = self.get_transformer()

        X, y = load_iris(return_X_y=True)
        
        Xt  = transfomer.fit_transform(X)
        self.assertIsNotNone(Xt, "Transformed set is none")
        self.assertIsInstance(Xt,type(X), "Wrong output type")
        self.assertTrue(Xt.shape[0] == X.shape[0], "Wrong number of rows")
        self.assertTrue(Xt.shape[1] == 1, "There should be only one colum!")

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






        