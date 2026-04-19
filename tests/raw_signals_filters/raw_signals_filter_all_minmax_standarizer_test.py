import unittest

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_minmax_standarizer import (
    RawSignalsFilterAllMinmaxStandarizer,
)

from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterAllStandarizerTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [
            RawSignalsFilterAllMinmaxStandarizer(),
            RawSignalsFilterAllMinmaxStandarizer(range_min=-1, range_max=1),
            RawSignalsFilterAllMinmaxStandarizer(range_min=-3, range_max=3),
        ]


if __name__ == "__main__":
    unittest.main()
