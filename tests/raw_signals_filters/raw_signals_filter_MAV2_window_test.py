from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_MAV2_window import RawSignalsFilterMAV2WindowFilter
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterAllPassTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterMAV2WindowFilter()]
        
