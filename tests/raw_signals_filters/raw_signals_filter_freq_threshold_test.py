import unittest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_freq_threshold import (
    RawSignalsFilterFreqThreshold,
)
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterFreqThresholdTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [
            RawSignalsFilterFreqThreshold(),
            RawSignalsFilterFreqThreshold(threshold=0.2),
            RawSignalsFilterFreqThreshold(threshold=0.5),
            ]


if __name__ == "__main__":
    unittest.main()
