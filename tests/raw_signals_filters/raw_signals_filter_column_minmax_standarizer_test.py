import unittest

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_column_minmax_standarizer import (
    RawSignalsFilterColumnMinMaxStandarizer,
)
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterColumnMinMaxStandarizerTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [
            RawSignalsFilterColumnMinMaxStandarizer(),
            RawSignalsFilterColumnMinMaxStandarizer(range_min=-1, range_max=1),
        ]


if __name__ == "__main__":
    unittest.main()
