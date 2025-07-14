import unittest

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsCreatorTest(unittest.TestCase):

    __test__ = False
    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise  unittest.SkipTest("Skipping")

    def get_creator(self,dtype=np.float32):
        raise  unittest.SkipTest("Skipping")

    def test_new_set(self):
        creator = self.get_creator()

        signals = creator.get_set()

        self.assertIsNotNone(signals, "Returned set is none.")
        self.assertIsInstance(signals, RawSignals)
        self.assertTrue( len(signals) >= 1, "Length must be greater or equall one.")

    def test_dtype(self):
        dtypes = [np.float32, np.float64, np.single, np.double]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                
                creator = self.get_creator(dtype=dtype)

                signals = creator.get_set()

                self.assertIsNotNone(signals, "Returned set is none.")
                self.assertIsInstance(signals, RawSignals)
                for sig in signals:
                    self.assertTrue(sig.to_numpy().dtype == dtype, "Wrong dtype of the signal")

if __name__ == '__main__':
    unittest.main()