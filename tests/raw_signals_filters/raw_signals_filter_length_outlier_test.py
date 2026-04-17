import unittest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_length_outlier import (
    RawSignalsFilterLengthOutlier,
)
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterLengthOutlierTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterLengthOutlier()]


if __name__ == "__main__":
    unittest.main()
