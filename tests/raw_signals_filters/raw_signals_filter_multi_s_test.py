import unittest

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_butter import (
    RawSignalsFilterButter,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_pass import (
    RawSignalsFilterAllPass,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_multi_s import RawSignalsFilterMultiSeq
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterMultiTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [
            RawSignalsFilterMultiSeq(
                filter_list=[
                    RawSignalsFilterButter(),
                    RawSignalsFilterButter(
                        low_freq=30, high_freq=450, btype="bandpass"
                    ),
                    RawSignalsFilterAllPass(),
                ]
            )
        ]


if __name__ == "__main__":
    unittest.main()
