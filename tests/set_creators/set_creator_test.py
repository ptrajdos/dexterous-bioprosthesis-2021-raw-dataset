import unittest
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import SetCreator
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_transformer_wrapper import SetCreatorTransformerWrapper
from tests.testing_tools import generate_sample_data
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class SetCreatorTest(unittest.TestCase):
    
    __test__ = False
    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise  unittest.SkipTest("Skipping")

    def get_creators(self)->SetCreator:
        raise  unittest.SkipTest("Skipping")
    
    def generate_sample_data(self):
        return generate_sample_data()

    def basic_test_check(self,raw_set,X,y,t):

        n_samples = len(raw_set)

        self.assertIsNotNone(X, "Dataset, object description is None")
        self.assertIsNotNone(y, "Class set is None")
        self.assertIsNotNone(t, "Timestamps is none")

        self.assertTrue(X.shape[0] == n_samples, "X -- wrong number of objects")
        self.assertTrue(len(y) == n_samples, "y -- wrong number of objects")
        self.assertTrue(len(t) == n_samples, "t -- wrong number of objects")

    def test_creator_fit_transform(self):

        creators = self.get_creators()

        for creator in creators:

            raw_set = self.generate_sample_data()
            n_samples = len(raw_set)

            X,y,t = creator.fit_transform(raw_set)

            self.basic_test_check(raw_set,X,y,t)

    def test_creator_fit_then_transform(self):

        creators = self.get_creators()

        for creator in creators:

            raw_set = self.generate_sample_data()
            n_samples = len(raw_set)

            creator.fit(raw_set)
            X,y,t = creator.transform(raw_set)

            self.basic_test_check(raw_set,X,y,t)


    def test_attributes_indices(self):
        creators = self.get_creators()

        for creator in creators:

            raw_set = self.generate_sample_data()
            n_samples = len(raw_set)
            n_channels = raw_set.signal_n_cols

            creator.fit(raw_set)
            X, y, t = creator.transform(raw_set)

            n_attribs = X.shape[1]

            indices = creator.get_channel_attribs_indices()

            if indices is not None:
                self.assertTrue( len(indices) == n_channels, "Wrong number of channels" )

                for channel_indices in indices:
                    try:
                        Xs = X[:,channel_indices]
                    except:
                        self.fail("Wrong channel indices")

                indices_coverage = [ 0 for _ in range(n_attribs)]

                for channel_indices in indices:
                    for index in channel_indices:
                        indices_coverage[index] += 1


                self.assertTrue(all([idx_count == 1 for idx_count in indices_coverage]), "Wrong coverage")

    def test_pipeline(self):
        creators = self.get_creators()

        for creator in creators:

            raw_set = self.generate_sample_data()
            
            pipeline = Pipeline([
                ('trans', SetCreatorTransformerWrapper(creator)), 
                ('classifier', DecisionTreeClassifier())
            ])

            y = raw_set.get_labels()
            pipeline.fit(raw_set,y)

            y_pred = pipeline.predict(raw_set)

    def test_gridsearch(self):
        creators = self.get_creators()

        for creator in creators:

            raw_set = self.generate_sample_data()
            
            pipeline = Pipeline([
                ('trans', SetCreatorTransformerWrapper(creator)), 
                ('classifier', DecisionTreeClassifier())
            ])
            params = [{
                "classifier__criterion":['gini', 'entropy' ]
            }]


            y = raw_set.get_labels()
            gs = GridSearchCV(pipeline, param_grid=params, scoring='accuracy',cv=3)

            gs.fit(raw_set,y)
            y_pred = gs.predict(raw_set)

            self.assertIsNotNone(y_pred, "Predictions are none")
            self.assertTrue( len(y) == len(y_pred), "Wrong predictions length")



if __name__ == '__main__':
    unittest.main()