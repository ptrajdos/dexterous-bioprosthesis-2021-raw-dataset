
import unittest
import numpy as np

class NpSignalExtractorTest(unittest.TestCase):

    __test__ = False
    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise  unittest.SkipTest("Skipping")

    def get_extractors(self):
        raise  unittest.SkipTest("Skipping")
    
    def test_basic(self):

        extractors = self.get_extractors()
        R = 100
        C =3
        X = np.random.random((R,C))

        for extractor in extractors:

            extractor.fit(X)
            att_per_column = extractor.attribs_per_column()

            n_desired_attrs = att_per_column * C

            T = extractor.transform(X)
            self.assertIsNotNone(T, "None type has been returned")
            self.assertIsInstance(T, np.ndarray, "Wrong type")
            self.assertTrue( len(T) == n_desired_attrs, "Wrong number of returned values" )
            self.assertFalse( np.any(np.isnan( T )), "NaNs in outut")
            self.assertTrue( np.all( np.isfinite( T) ), "Infinite values in output" )

    def test_fittransform(self):

        extractors = self.get_extractors()
        R = 100
        C =3
        X = np.random.random((R,C))

        for extractor in extractors:

            T = extractor.fit_transform(X)

            att_per_column = extractor.attribs_per_column()
            n_desired_attrs = att_per_column * C

            self.assertIsNotNone(T, "None type has been returned")
            self.assertIsInstance(T, np.ndarray, "Wrong type")
            self.assertTrue( len(T) == n_desired_attrs, "Wrong number of returned values" )
            self.assertFalse( np.any(np.isnan( T )), "NaNs in outut")
            self.assertTrue( np.all( np.isfinite( T) ), "Infinite values in output" )

        


