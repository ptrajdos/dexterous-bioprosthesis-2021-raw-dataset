
import unittest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_pass import RawSignalsFilterAllPass
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterAllPassTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterAllPass()]
        
if __name__ == '__main__':
    unittest.main()