import unittest

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_robuts_standarizer import (
    RawSignalsFilterAllRobustStandarizer,
)

from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterAllStandarizerTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterAllRobustStandarizer()]


if __name__ == "__main__":
    unittest.main()
