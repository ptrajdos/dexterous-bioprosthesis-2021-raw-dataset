
import unittest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_single import RawSignalsFilterWindowSingle
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterWindowSingleTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterWindowSingle(start_sample=0, end_sample=15)]
    
    def test_window_cut(self):
        
        sig_filters = self.get_filters()

        n_samples = 100
        signals = self.generate_sample_data(samples_number=n_samples)

        for sig_filter in sig_filters:

            f_signals = sig_filter.fit_transform(signals)

            for signal in f_signals:
                np_sig = signal.signal
                self.assertTrue( np_sig.shape[0] == 15, "Wrong length of windowed signal" )


        
if __name__ == '__main__':
    unittest.main()