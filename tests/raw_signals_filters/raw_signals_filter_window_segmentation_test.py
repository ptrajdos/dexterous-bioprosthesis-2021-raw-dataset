import unittest

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_segmentation import (
    RawSignalsFilterWindowSegmentation,
)
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterWindowSegmentationTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterWindowSegmentation(window_length=10, overlap=5), 
                RawSignalsFilterWindowSegmentation(window_length=0.33, overlap=0.5)]

    def test_windowing(self):

        sig_filters = [RawSignalsFilterWindowSegmentation(window_length=10, overlap=5)]

        n_samples = 30
        signals = self.generate_sample_data(samples_number=n_samples)
        signals_len = len(signals)

        for sig_filter in sig_filters:

            f_signals = sig_filter.fit_transform(signals)
            f_signals_len = len(f_signals)

            self.assertTrue(f_signals_len == 5 * signals_len)

    def test_windowing_max_overlap(self):
        """Overlap of window_length - 1 should produce maximum number of windows."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=9)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((30, 2))))

        f_signals = sig_filter.fit_transform(signals)
        # step = 10 - 9 = 1, windows: start 0..20 → 21 windows
        self.assertEqual(len(f_signals), 21)

    def test_windowing_overlap_equals_one(self):
        """Minimal overlap of 1 sample should produce maximum number of windows."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=1)
        n_samples = 30
        n_channels = 3
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((n_samples, n_channels))))

        f_signals = sig_filter.fit_transform(signals)
        # step = 10 - 1 = 9, windows: start at 0,9,18 → end at 10,19,28. 
        # Next start=27, end=37 > 30 → 3 windows
        self.assertEqual(len(f_signals), 3)

    def test_windowing_window_shape(self):
        """Each output window should have the correct shape."""
        window_length = 10
        n_channels = 4
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=window_length, overlap=5)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((30, n_channels))))

        f_signals = sig_filter.fit_transform(signals)

        for sig in f_signals:
            self.assertEqual(sig.signal.shape[0], window_length)
            self.assertEqual(sig.signal.shape[1], n_channels)

    def test_windowing_data_integrity(self):
        """Window content should match the corresponding slice of the original signal."""
        window_length = 10
        overlap = 5
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=window_length, overlap=overlap)
        original_data = np.arange(60).reshape(30, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)

        step = window_length - overlap  # 5
        for i, sig in enumerate(f_signals):
            start = i * step
            expected = original_data[start:start + window_length, :]
            np.testing.assert_array_equal(sig.signal, expected)

    def test_windowing_preserves_labels(self):
        """All windows from a signal should inherit the original signal's label."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=5)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((30, 3)), object_class="classA"))
        signals.append(RawSignal(signal=np.random.random((30, 3)), object_class="classB"))

        f_signals = sig_filter.fit_transform(signals)

        labels = f_signals.get_labels()
        count_a = np.sum(labels == "classA")
        count_b = np.sum(labels == "classB")
        # Each 30-sample signal with window=10, overlap=5 → 5 windows
        self.assertEqual(count_a, 5)
        self.assertEqual(count_b, 5)

    def test_windowing_float_params(self):
        """Float window_length and overlap should produce correct number of windows."""
        # window_length=0.5 of 20 samples → 10, overlap=0.5 of 10 → 5, step=5
        # windows: 0-10, 5-15, 10-20 → 3 windows
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=0.5, overlap=0.5)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(len(f_signals), 3)

    def test_windowing_float_window_shape(self):
        """Float params should produce windows with correct effective length."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=0.5, overlap=0.5)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 3))))

        f_signals = sig_filter.fit_transform(signals)
        for sig in f_signals:
            self.assertEqual(sig.signal.shape[0], 10)  # 0.5 * 20 = 10
            self.assertEqual(sig.signal.shape[1], 3)

    def test_windowing_multiple_signals(self):
        """Filter should process each signal independently."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=5)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((30, 2))))  # 5 windows
        signals.append(RawSignal(signal=np.random.random((20, 2))))  # 3 windows
        signals.append(RawSignal(signal=np.random.random((10, 2))))  # 1 window

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(len(f_signals), 9)

    def test_windowing_exact_fit(self):
        """Signal length exactly divisible by step should use all samples."""
        # window=10, overlap=5, step=5, signal=20 → windows at 0,5,10 → end 10,15,20 → 3 windows? 
        # start=0→end=10, start=5→end=15, start=10→end=20 → 3 windows (end<=20)
        # But wait: also start=15→end=25>20 → stop. So 3 windows? 
        # Actually from test_windowing: 30 samples → 5 windows. Let's verify: 
        # 0-10,5-15,10-20,15-25,20-30 → 5. For 20: 0-10,5-15,10-20 → 3.
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=5)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(len(f_signals), 3)

    def test_windowing_signal_equals_window(self):
        """When signal length equals window length, exactly one window should be produced."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=5)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((10, 3))))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(len(f_signals), 1)

    def test_windowing_signal_shorter_than_window(self):
        """When signal is shorter than window, a ValueError should be raised."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=5)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((8, 3))))

        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_windowing_empty_signals(self):
        """Filtering empty RawSignals should return empty RawSignals."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=5)
        signals = RawSignals()

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(len(f_signals), 0)
        self.assertIsInstance(f_signals, RawSignals)

    def test_windowing_preserves_channel_names(self):
        """Windows should preserve channel names from the original signal."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=5)
        channel_names = ["ch1", "ch2", "ch3"]
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((30, 3)), object_class="a", channel_names=channel_names))

        f_signals = sig_filter.fit_transform(signals)
        for sig in f_signals:
            self.assertEqual(list(sig.channel_names), channel_names)

    def test_windowing_preserves_timestamp(self):
        """Windows should preserve the timestamp from the original signal."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=5)
        timestamp = 1234567890.0
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((30, 3)), object_class="a", timestamp=timestamp))

        f_signals = sig_filter.fit_transform(signals)
        for sig in f_signals:
            self.assertEqual(sig.timestamp, timestamp)

    def test_windowing_single_channel(self):
        """Windowing should work correctly with single-channel signals."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=5, overlap=2)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((15, 1))))

        f_signals = sig_filter.fit_transform(signals)
        # step = 5 - 2 = 3, windows: 0-5, 3-8, 6-11, 9-14 → end 14 <=15? No, end=14 means [9:14] which is 5 samples
        # Actually end_idx = start + 5: 0→5, 3→8, 6→11, 9→14, 12→17>15 → 4 windows
        self.assertEqual(len(f_signals), 4)
        for sig in f_signals:
            self.assertEqual(sig.signal.shape, (5, 1))

    def test_wrong_args(self):
        n_samples = 30
        signals = self.generate_sample_data(samples_number=n_samples)

        wrong_args = {
            "negative window_length int": RawSignalsFilterWindowSegmentation(window_length=-1, overlap=5),
            "negative overlap int": RawSignalsFilterWindowSegmentation(window_length=5, overlap=-1),
            "window_length too big int": RawSignalsFilterWindowSegmentation(window_length=n_samples+1, overlap=5),
            "overlap bigger than window_length": RawSignalsFilterWindowSegmentation(window_length=10, overlap=15),
            "overlap bigger than n_samples": RawSignalsFilterWindowSegmentation(window_length=10, overlap=n_samples+1),

            "negative window_length float": RawSignalsFilterWindowSegmentation(window_length=-1.0, overlap=0.5),
            "negative overlap float": RawSignalsFilterWindowSegmentation(window_length=0.4, overlap=-1.0),
            "out of range window_length float": RawSignalsFilterWindowSegmentation(window_length=1.1, overlap=0.3),
            "out of range overlap float": RawSignalsFilterWindowSegmentation(window_length=0.45, overlap=1.3),

            "wrong type window_length": RawSignalsFilterWindowSegmentation(window_length="ax", overlap=3),
            "wrong type overlap": RawSignalsFilterWindowSegmentation(window_length=5, overlap="ax"),
        }
        for filter_name, filter in wrong_args.items():
            with self.subTest(filter_name=filter_name):
                with self.assertRaises(ValueError):
                    f_signals = filter.fit_transform(signals)

    def test_wrong_args_mixed_types(self):
        """Mixing int and float types for window_length and overlap should raise ValueError."""
        n_samples = 30
        signals = self.generate_sample_data(samples_number=n_samples)

        mixed_args = {
            "int window_length float overlap": RawSignalsFilterWindowSegmentation(window_length=10, overlap=0.5),
            "float window_length int overlap": RawSignalsFilterWindowSegmentation(window_length=0.5, overlap=5),
        }
        for filter_name, sig_filter in mixed_args.items():
            with self.subTest(filter_name=filter_name):
                with self.assertRaises(ValueError):
                    sig_filter.fit_transform(signals)

    def test_wrong_args_zero_window_length_int(self):
        """Zero window_length (int) should raise ValueError."""
        signals = self.generate_sample_data(samples_number=30)
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=0, overlap=0)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_zero_overlap_int(self):
        """Zero overlap (int) should raise ValueError."""
        signals = self.generate_sample_data(samples_number=30)
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=0)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_zero_window_length_float(self):
        """Zero window_length (float) should raise ValueError."""
        signals = self.generate_sample_data(samples_number=30)
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=0.0, overlap=0.5)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_zero_overlap_float(self):
        """Zero overlap (float) should raise ValueError."""
        signals = self.generate_sample_data(samples_number=30)
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=0.5, overlap=0.0)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_boundary_float(self):
        """Float values exactly at 1.0 should raise ValueError."""
        signals = self.generate_sample_data(samples_number=30)

        boundary_args = {
            "window_length exactly 1.0": RawSignalsFilterWindowSegmentation(window_length=1.0, overlap=0.5),
            "overlap exactly 1.0": RawSignalsFilterWindowSegmentation(window_length=0.5, overlap=1.0),
        }
        for filter_name, sig_filter in boundary_args.items():
            with self.subTest(filter_name=filter_name):
                with self.assertRaises(ValueError):
                    sig_filter.fit_transform(signals)

    def test_wrong_args_none_type(self):
        """None values for parameters should raise ValueError."""
        signals = self.generate_sample_data(samples_number=30)

        none_args = {
            "None window_length": RawSignalsFilterWindowSegmentation(window_length=None, overlap=5),
            "None overlap": RawSignalsFilterWindowSegmentation(window_length=5, overlap=None),
        }
        for filter_name, sig_filter in none_args.items():
            with self.subTest(filter_name=filter_name):
                with self.assertRaises(ValueError):
                    sig_filter.fit_transform(signals)

    def test_windowing_sample_rate_preserved(self):
        """Output RawSignals should preserve the sample rate."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=5)
        signals = RawSignals(sample_rate=500)
        signals.append(RawSignal(signal=np.random.random((30, 3))))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals.get_sample_rate(), 500)

    def test_windowing_independent_copies(self):
        """Modifying a window's data should not affect other windows or the original."""
        sig_filter = RawSignalsFilterWindowSegmentation(window_length=10, overlap=5)
        original_data = np.ones((20, 2))
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)

        # Modify first window
        f_signals[0].signal[:] = 999.0

        # Other windows should be unaffected
        for i in range(1, len(f_signals)):
            self.assertFalse(np.any(f_signals[i].signal == 999.0))

if __name__ == "__main__":
    unittest.main()
