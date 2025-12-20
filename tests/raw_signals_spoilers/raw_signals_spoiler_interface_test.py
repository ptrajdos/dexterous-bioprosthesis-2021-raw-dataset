import unittest

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals

class RawSignalsSpoilerInterfaceTest(unittest.TestCase):

    __test__ = False
    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise  unittest.SkipTest("Skipping")
        
    def get_spoilers(self):
        """
        gets the spoilers to be tested
        """
        raise  unittest.SkipTest("Skipping")
    
    def generate_sample_data(self,signal_number=10, column_number=3, samples_number=12, dtype=np.double)->RawSignals:
        signals = RawSignals()

        for i in range(1,signal_number+1):
            signals.append( RawSignal(signal=np.random.random( (samples_number,column_number)).astype(dtype)) )

        return signals

    def generate_zero_data(self,signal_number=10, column_number=3, samples_number=12, dtype=np.double)->RawSignals:
        signals = RawSignals()

        for i in range(1,signal_number+1):
            signals.append( RawSignal(signal=np.zeros( (samples_number,column_number)).astype(dtype)) )

        return signals
    
    def _check_for_invalid_values(self,raw_signals:RawSignals):

        for signal in raw_signals:
            self.assertFalse(  np.any(np.isnan(signal.to_numpy())), "NaNs in transformed data" )
            self.assertFalse(  np.any(np.isinf(signal.to_numpy())), "Infs in transformed data" )
    
    def test_fit_then_transform(self):

        data = self.generate_sample_data()

        spoilers = self.get_spoilers()
        rcs = ((10,1,20), (5,3,15), (8,4,12), (12,2,25))

        for N,C,R in rcs:
            data = self.generate_sample_data(signal_number=N, column_number=C, samples_number=R)
            for spoiler in spoilers:
                with self.subTest(spoiler=spoiler, N=N, C=C, R=R):
                    spoiler.fit(data)
                    t_data = spoiler.transform(data)

                    self.assertIsNotNone(t_data, "Transformed data is None")
                    self.assertIsInstance(t_data, RawSignals, "Transformed data is not an instance of RawSignals")
                    self._check_for_invalid_values(t_data)

    def test_fit_transform(self):

        data = self.generate_sample_data()

        spoilers = self.get_spoilers()

        for spoiler in spoilers:

            t_data = spoiler.fit_transform(data)

            self.assertIsNotNone(t_data, "Transformed data is None")
            self.assertIsInstance(t_data, RawSignals, "Transformed data is not an instance of RawSignals")
            self._check_for_invalid_values(t_data)

    def test_fit_transform_zero(self):

        data = self.generate_zero_data()

        spoilers = self.get_spoilers()

        for spoiler in spoilers:

            t_data = spoiler.fit_transform(data)

            self.assertIsNotNone(t_data, "Transformed data is None")
            self.assertIsInstance(t_data, RawSignals, "Transformed data is not an instance of RawSignals")
            self._check_for_invalid_values(t_data)