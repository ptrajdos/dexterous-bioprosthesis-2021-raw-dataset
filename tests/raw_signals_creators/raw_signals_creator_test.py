import unittest

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsCreatorTest(unittest.TestCase):

    __test__ = False
    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise  unittest.SkipTest("Skipping")

    def get_creator(self):
        raise  unittest.SkipTest("Skipping")

    def test_new_set(self):
        creator = self.get_creator()

        signals = creator.get_set()

        self.assertIsNotNone(signals, "Returned set is none.")
        self.assertIsInstance(signals, RawSignals)
        self.assertTrue( len(signals) >= 1, "Length must be greater or equall one.")

if __name__ == '__main__':
    unittest.main()