import unittest
from dexterous_bioprosthesis_2021_raw_datasets.model_selection.class_combination_generator import ClassCombinationGenerator
from sklearn.datasets import load_iris
import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import RawSignalsCreatorSines
from sklearn.exceptions import NotFittedError

class ClassCombinationGeneratorTest(unittest.TestCase):

    def test_combinations(self):

        gen = ClassCombinationGenerator()

        X, y = load_iris(return_X_y=True)
        n_classes =  len( np.unique(y))

        gen.fit(X,y)

        counter =0
        for indices in gen.get_indices():
            counter +=1
            _ = X[indices]
            _ = y[indices]

        self.assertTrue( counter == gen.n_combinations, "Wrong number of combinations")

    def test_combinations_on_raw_dataset(self):

        gen = ClassCombinationGenerator()

        data_creator  = RawSignalsCreatorSines()

        raw_set = data_creator.get_set()
        y = raw_set.get_labels()

        gen.fit(raw_set, y)

        counter =0
        for indices in gen.get_indices():
            counter +=1

            _ = raw_set[indices]
            _ = y[indices]

        self.assertTrue( counter == gen.n_combinations, "Wrong number of combinations")

    def test_not_fitted(self):

        gen = ClassCombinationGenerator()

        try:
            for i in gen.get_indices():
                pass
            self.fail("No not fitted exception has been raised")
        except NotFittedError as nf:
            self.assertTrue(True)
        except Exception as ex: 
            self.fail("Exception other than not fitted exception has been raised: {}".format(ex))

if __name__ == '__main__':
    unittest.main()