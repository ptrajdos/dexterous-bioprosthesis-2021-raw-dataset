import unittest

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_column_robust_standarizer import (
    RawSignalsFilterColumnRobustStandarizer,
)
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterColumnStandarizerTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterColumnRobustStandarizer()]


if __name__ == "__main__":
    unittest.main()
