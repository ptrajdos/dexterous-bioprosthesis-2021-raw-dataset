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

    def test_timestamps(self):
        n_samples = 30
        signals = self.generate_sample_data(samples_number=n_samples)
        fs = signals.sample_rate
        base_timestamp = 1000.0
        for signal in signals:
            signal.timestamp = base_timestamp

        window_length = 10
        overlap = 5
        sig_filter = RawSignalsFilterWindowSegmentationFS(
            window_length=window_length, overlap=overlap
        )
        f_signals = sig_filter.fit_transform(signals)

        step_samples = int(window_length * fs / 1000) - int(overlap * fs / 1000)
        windows_per_signal = len(f_signals) // len(signals)
        for i, f_signal in enumerate(f_signals):
            window_idx = i % windows_per_signal
            expected = base_timestamp + window_idx * step_samples / fs
            self.assertAlmostEqual(f_signal.timestamp, expected)

        # Original signals must not be modified.
        for signal in signals:
            self.assertEqual(signal.timestamp, base_timestamp)


if __name__ == "__main__":
    unittest.main()
