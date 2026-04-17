import unittest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_length_median_standarizer import (
    RawSignalsFilterLengthMedianStandarizer,
)
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterLengthMedianStandarizerTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterLengthMedianStandarizer()]


if __name__ == "__main__":
    unittest.main()
