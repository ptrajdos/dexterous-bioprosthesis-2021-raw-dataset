import abc
import unittest

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals

class RawSignalsFilterTest(unittest.TestCase):

    __test__ = False
    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise  unittest.SkipTest("Skipping")

    @abc.abstractclassmethod
    def get_filters(self):
        """
        gets the filter to be tested
        """
        raise  unittest.SkipTest("Skipping")

    def generate_sample_data(self,signal_number=10, column_number=3, samples_number=12)->RawSignals:
        signals = RawSignals()

        for i in range(1,signal_number+1):
            signals.append( RawSignal(signal=np.random.random( (samples_number,column_number))) )

        return signals


    def test_filter_fit_then_transform(self):
        filters = self.get_filters()
        signals = RawSignals()
        N = 10
        M=30
        C = 6
        for filter in filters:

            for i in range(1,N+1):
                signals.append( RawSignal(signal=np.random.random( (M*i,C))) )
            

            try:
                filter.fit(signals)
                tr_signals = filter.transform(signals)

                self.assertIsNotNone(tr_signals,"Transformed object is none.")
                self.assertIsInstance(tr_signals, RawSignals)
                self.assertTrue(id(signals) != id(tr_signals), "Objects are the same")

            except Exception as ex:
                self.fail("An exception has been caught: {}".format(ex))

    def test_fit_transform(self):
        filters = self.get_filters()
        signals = self.generate_sample_data(samples_number=30)
        for filter in filters:
            try:
                tr_signals = filter.fit_transform(signals)

                self.assertIsNotNone(tr_signals,"Transformed object is none.")
                self.assertIsInstance(tr_signals, RawSignals)
                self.assertTrue(id(signals) != id(tr_signals), "Objects are the same")

            except Exception as ex:
                self.fail("An exception has been caught: {}".format(ex))


if __name__ == '__main__':
    unittest.main()