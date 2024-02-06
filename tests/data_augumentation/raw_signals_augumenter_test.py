import unittest

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import RawSignalsCreatorSines

class RawSignalsAugumenterTest(unittest.TestCase):

    __test__ = False
    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise  unittest.SkipTest("Skipping")

    def get_augumenter(self):
       raise  unittest.SkipTest("Skipping")

    def test_fit_then_transform(self):
        signal_creator = RawSignalsCreatorSines(samples_number=1000)
        raw_signals = signal_creator.get_set()

        aug = self.get_augumenter()

        obj = aug.fit(raw_signals)
        self.assertIsNotNone( obj,  "fit should return something")
        self.assertTrue(type(obj) == type (aug), "Fit should return self")
        aug_signals = aug.transform(raw_signals)

        self.assertIsNotNone( aug_signals, "Augumented signals none." )
        self.assertTrue( len(aug_signals) >= 1, "Number of augumented points.")


    def test_fit_transform(self):
        signal_creator = RawSignalsCreatorSines()
        raw_signals = signal_creator.get_set()

        aug = self.get_augumenter()
       
        aug_signals = aug.fit_transform(raw_signals)

        self.assertIsNotNone( aug_signals, "Augumented signals none." )
        self.assertTrue( len(aug_signals) >= 1, "Number of augumented points.")




if __name__ == '__main__':
    unittest.main()