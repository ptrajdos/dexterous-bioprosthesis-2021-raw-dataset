
import unittest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_pass import RawSignalsFilterAllPass
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_obj_resampler import RawSignalsFilterObjResampler
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterObjResamplerTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [
            RawSignalsFilterObjResampler(),
            RawSignalsFilterObjResampler(resampling_rate=0.5, with_replacement=False),
            RawSignalsFilterObjResampler(resampling_rate=2.0, with_replacement=True)
            ]

    def test_wrong_config(self):
        with self.assertRaises(ValueError):
            filter = RawSignalsFilterObjResampler(resampling_rate=2.0, with_replacement=False)
            signals = self.generate_sample_data(signal_number=5, samples_number=10)
            filter.transform(signals)
        
if __name__ == '__main__':
    unittest.main()