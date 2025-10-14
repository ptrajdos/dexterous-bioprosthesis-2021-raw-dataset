import unittest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_segmentation_fs import (
    RawSignalsFilterWindowSegmentationFS,
)
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterWindowSegmentationFSTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterWindowSegmentationFS(window_length=10, overlap=5)]

    def test_windowing(self):

        sig_filters = self.get_filters()

        n_samples = 30
        signals = self.generate_sample_data(samples_number=n_samples)
        signals_len = len(signals)

        for sig_filter in sig_filters:

            f_signals = sig_filter.fit_transform(signals)
            f_signals_len = len(f_signals)

            self.assertTrue(f_signals_len == 5 * signals_len)


if __name__ == "__main__":
    unittest.main()
