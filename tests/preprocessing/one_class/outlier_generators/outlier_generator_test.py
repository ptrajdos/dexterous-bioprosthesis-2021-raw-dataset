import unittest
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
import numpy as np

class OutlierGeneratorTest(unittest.TestCase):

    __test__ = False
    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise  unittest.SkipTest("Skipping")

    def get_generators(self):
        return None
    

    def test_generation(self):

        X, y = load_iris(return_X_y=True)

        generators = self.get_generators()

        for generator in generators:

            try:
                generator.generate()
                self.fail("Not fitted object should have risen NotFittedError exception")
            except NotFittedError:
                pass
            except:
                self.fail("Wrong exception type generated for unfitted object")

            generator.fit(X,y)

            Xg, yg = generator.generate()

            self.assertIsNotNone(Xg, "Generated X is none")
            self.assertIsNotNone(yg, "Generated y is none")
            self.assertIsInstance(Xg, np.ndarray , "generated X is not numpy array")
            self.assertIsInstance(yg, np.ndarray , "generated y is not numpy array")
            self.assertTrue(len(Xg)==len(yg), "Different number of objects in X and y")
            self.assertFalse( np.any(np.isinf(Xg)), "Infinite values in generated X")
            self.assertFalse( np.any(np.isinf(yg)), "Infinite values in generated y")
            self.assertTrue( y.dtype == yg.dtype, "Wrong array dtypes" )

    def test_fit_generation(self):

        X, y = load_iris(return_X_y=True)

        generators = self.get_generators()

        for generator in generators:
    
            generator.fit(X,y)

            Xg, yg = generator.fit_generate(X,y)

            self.assertIsNotNone(Xg, "Generated X is none")
            self.assertIsNotNone(yg, "Generated y is none")
            self.assertIsInstance(Xg, np.ndarray , "generated X is not numpy array")
            self.assertIsInstance(yg, np.ndarray , "generated y is not numpy array")
            self.assertTrue(len(Xg)==len(yg), "Different number of objects in X and y")
            self.assertFalse( np.any(np.isinf(Xg)), "Infinite values in generated X")
            self.assertFalse( np.any(np.isinf(yg)), "Infinite values in generated y")
            self.assertTrue( y.dtype == yg.dtype, "Wrong array dtypes" )
