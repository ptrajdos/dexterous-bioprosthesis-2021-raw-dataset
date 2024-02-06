
import unittest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_pass import RawSignalsFilterAllPass
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_decimate import RawSignalsFilterDecimate
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterDecimateTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterDecimate()]
    
    def test_decimation(self):
        
        sig_filters = self.get_filters()
        for sig_filter in sig_filters:

            n_samples = 100
            signals = self.generate_sample_data(samples_number=n_samples)

            s_freq = signals.sample_rate

            f_signals = sig_filter.fit_transform(signals)

            self.assertTrue( f_signals.sample_rate == s_freq//2, "Wrong sample frequency after decimation." )

            for signal in f_signals:
                np_sig = signal.signal
                self.assertTrue( np_sig.shape[0] == n_samples//2, "Wrong length of decimated signal" )


        
if __name__ == '__main__':
    unittest.main()