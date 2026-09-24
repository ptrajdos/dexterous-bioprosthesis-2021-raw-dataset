import unittest

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_column_quatile_standarizer import (
    RawSignalsFilterQuantileStandarizer,
)
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterQuantileStandarizerTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterQuantileStandarizer(),
                RawSignalsFilterQuantileStandarizer(q_low=0.1, q_high=0.9),
                RawSignalsFilterQuantileStandarizer(q_low=0.25, q_high=0.75),
                RawSignalsFilterQuantileStandarizer(clip=True)
                ]


if __name__ == "__main__":
    unittest.main()
