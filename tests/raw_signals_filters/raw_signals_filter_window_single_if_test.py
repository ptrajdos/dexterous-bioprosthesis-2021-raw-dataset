import unittest

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_single_if import (
    RawSignalsFilterWindowSingleIF,
)
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterWindowSingleIFTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [
            RawSignalsFilterWindowSingleIF(offset=0, length=10),
            RawSignalsFilterWindowSingleIF(offset=5, length=10),
            RawSignalsFilterWindowSingleIF(offset=0.0, length=0.5),
            RawSignalsFilterWindowSingleIF(offset=0.25, length=0.5),
            RawSignalsFilterWindowSingleIF(offset=5, length=0.5),
            RawSignalsFilterWindowSingleIF(offset=0.25, length=5),
            RawSignalsFilterWindowSingleIF(offset=0, length=None),
            RawSignalsFilterWindowSingleIF(offset=5, length=None),
            RawSignalsFilterWindowSingleIF(offset=0.25, length=None),
        ]

    # --- Integer parameter tests ---

    def test_int_basic_extraction(self):
        """Integer offset and length should extract the correct window."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=5, length=10)
        original_data = np.arange(40).reshape(20, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(len(f_signals), 1)
        self.assertEqual(f_signals[0].signal.shape, (10, 2))
        np.testing.assert_array_equal(f_signals[0].signal, original_data[5:15, :])

    def test_int_offset_zero(self):
        """Offset=0 should extract from the beginning of the signal."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0, length=10)
        original_data = np.arange(40).reshape(20, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        np.testing.assert_array_equal(f_signals[0].signal, original_data[0:10, :])

    def test_int_extract_entire_signal(self):
        """Offset=0 and length=signal_length should extract the entire signal."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0, length=20)
        original_data = np.arange(40).reshape(20, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals[0].signal.shape, (20, 2))
        np.testing.assert_array_equal(f_signals[0].signal, original_data)

    def test_int_extract_last_samples(self):
        """Extracting the last samples of the signal should work correctly."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=15, length=5)
        original_data = np.arange(40).reshape(20, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        np.testing.assert_array_equal(f_signals[0].signal, original_data[15:20, :])

    def test_int_length_one(self):
        """Length=1 should extract a single sample."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=5, length=1)
        original_data = np.arange(20).reshape(10, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals[0].signal.shape, (1, 2))
        np.testing.assert_array_equal(f_signals[0].signal, original_data[5:6, :])

    def test_int_multiple_signals(self):
        """Filter should process each signal independently with integer params."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=2, length=5)
        signals = RawSignals()
        data1 = np.ones((20, 2)) * 1.0
        data2 = np.ones((20, 2)) * 2.0
        data3 = np.ones((20, 2)) * 3.0
        signals.append(RawSignal(signal=data1))
        signals.append(RawSignal(signal=data2))
        signals.append(RawSignal(signal=data3))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(len(f_signals), 3)
        np.testing.assert_array_equal(f_signals[0].signal, np.ones((5, 2)) * 1.0)
        np.testing.assert_array_equal(f_signals[1].signal, np.ones((5, 2)) * 2.0)
        np.testing.assert_array_equal(f_signals[2].signal, np.ones((5, 2)) * 3.0)

    def test_int_single_channel(self):
        """Integer params should work correctly with single-channel signals."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=3, length=5)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((15, 1))))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals[0].signal.shape, (5, 1))

    # --- Float parameter tests ---

    def test_float_basic_extraction(self):
        """Float offset and length should extract the correct window."""
        # offset=0.25 of 20 → 5, length=0.5 of 20 → 10
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0.25, length=0.5)
        original_data = np.arange(40).reshape(20, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals[0].signal.shape, (10, 2))
        np.testing.assert_array_equal(f_signals[0].signal, original_data[5:15, :])

    def test_float_offset_zero(self):
        """Float offset=0.0 should extract from the beginning."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0.0, length=0.5)
        original_data = np.arange(40).reshape(20, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        # effective_offset = round(0.0 * 20) = 0, effective_length = round(0.5 * 20) = 10
        self.assertEqual(f_signals[0].signal.shape, (10, 2))
        np.testing.assert_array_equal(f_signals[0].signal, original_data[0:10, :])

    def test_float_window_shape(self):
        """Float params should produce windows with correct effective shape."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0.1, length=0.5)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 3))))

        f_signals = sig_filter.fit_transform(signals)
        # effective_offset = round(0.1 * 20) = 2, effective_length = round(0.5 * 20) = 10
        self.assertEqual(f_signals[0].signal.shape[0], 10)
        self.assertEqual(f_signals[0].signal.shape[1], 3)

    def test_float_multiple_signals(self):
        """Float params should process each signal independently."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0.0, length=0.5)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))
        signals.append(RawSignal(signal=np.random.random((40, 2))))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(len(f_signals), 2)
        self.assertEqual(f_signals[0].signal.shape[0], 10)  # 0.5 * 20
        self.assertEqual(f_signals[1].signal.shape[0], 20)  # 0.5 * 40

    # --- Metadata preservation tests ---

    def test_preserves_labels(self):
        """Output signals should preserve the original signal's label."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0, length=10)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2)), object_class="classA"))
        signals.append(RawSignal(signal=np.random.random((20, 2)), object_class="classB"))

        f_signals = sig_filter.fit_transform(signals)
        labels = f_signals.get_labels()
        self.assertEqual(labels[0], "classA")
        self.assertEqual(labels[1], "classB")

    def test_preserves_channel_names(self):
        """Output signals should preserve channel names."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0, length=10)
        channel_names = ["ch1", "ch2", "ch3"]
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 3)), channel_names=channel_names))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(list(f_signals[0].channel_names), channel_names)

    def test_preserves_timestamp(self):
        """Output signals should preserve the timestamp."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0, length=10)
        timestamp = 1234567890.0
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2)), timestamp=timestamp))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals[0].timestamp, timestamp)

    def test_preserves_sample_rate(self):
        """Output RawSignals should preserve the sample rate."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0, length=10)
        signals = RawSignals(sample_rate=500)
        signals.append(RawSignal(signal=np.random.random((20, 2))))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals.get_sample_rate(), 500)

    # --- Deep copy / independence tests ---

    def test_independent_copy(self):
        """Modifying the output should not affect the original signal."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0, length=10)
        original_data = np.ones((20, 2))
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        f_signals[0].signal[:] = 999.0

        np.testing.assert_array_equal(signals[0].signal, np.ones((20, 2)))

    def test_empty_signals(self):
        """Filtering empty RawSignals should return empty RawSignals."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0, length=10)
        signals = RawSignals()

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(len(f_signals), 0)
        self.assertIsInstance(f_signals, RawSignals)

    # --- Validation / error tests ---

    def test_wrong_args_negative_offset_int(self):
        """Negative integer offset should raise ValueError."""
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))
        sig_filter = RawSignalsFilterWindowSingleIF(offset=-1, length=10)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_zero_length_int(self):
        """Zero integer length should raise ValueError."""
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0, length=0)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_offset_exceeds_signal_int(self):
        """Integer offset >= signal length should raise ValueError."""
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))
        sig_filter = RawSignalsFilterWindowSingleIF(offset=20, length=5)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_offset_plus_length_exceeds_signal_int(self):
        """Integer offset + length > signal length should raise ValueError."""
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))
        sig_filter = RawSignalsFilterWindowSingleIF(offset=15, length=10)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_float_offset_negative(self):
        """Negative float offset should raise ValueError."""
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))
        sig_filter = RawSignalsFilterWindowSingleIF(offset=-0.1, length=0.5)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_float_offset_at_one(self):
        """Float offset exactly 1.0 should raise ValueError."""
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))
        sig_filter = RawSignalsFilterWindowSingleIF(offset=1.0, length=0.5)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_float_length_zero(self):
        """Float length 0.0 should raise ValueError."""
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0.0, length=0.0)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_float_length_at_one(self):
        """Float length exactly 1.0 should raise ValueError (offset+length > 1 when offset > 0)."""
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0.0, length=1.0)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_wrong_args_float_sum_exceeds_one(self):
        """Float offset + length > 1.0 should raise ValueError."""
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0.6, length=0.5)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    def test_mixed_int_offset_float_length(self):
        """Int offset with float length should work correctly."""
        # offset=5 (absolute), length=0.5 of 20 → 10
        sig_filter = RawSignalsFilterWindowSingleIF(offset=5, length=0.5)
        original_data = np.arange(40).reshape(20, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals[0].signal.shape, (10, 2))
        np.testing.assert_array_equal(f_signals[0].signal, original_data[5:15, :])

    def test_mixed_float_offset_int_length(self):
        """Float offset with int length should work correctly."""
        # offset=0.25 of 20 → 5, length=10 (absolute)
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0.25, length=10)
        original_data = np.arange(40).reshape(20, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals[0].signal.shape, (10, 2))
        np.testing.assert_array_equal(f_signals[0].signal, original_data[5:15, :])

    def test_mixed_types_validation_errors(self):
        """Mixed types should still validate each parameter independently."""
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))

        # int offset negative + float length
        with self.assertRaises(ValueError):
            RawSignalsFilterWindowSingleIF(offset=-1, length=0.5).fit_transform(signals)

        # float offset invalid + int length
        with self.assertRaises(ValueError):
            RawSignalsFilterWindowSingleIF(offset=1.0, length=5).fit_transform(signals)

        # int offset + float length exceeding signal
        with self.assertRaises(ValueError):
            RawSignalsFilterWindowSingleIF(offset=15, length=0.5).fit_transform(signals)

    def test_wrong_args_none_offset(self):
        """None offset should raise ValueError."""
        signals = RawSignals()
        signals.append(RawSignal(signal=np.random.random((20, 2))))
        sig_filter = RawSignalsFilterWindowSingleIF(offset=None, length=5)
        with self.assertRaises(ValueError):
            sig_filter.fit_transform(signals)

    # --- None length tests ---

    def test_none_length_from_start(self):
        """length=None with offset=0 should return the entire signal."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0, length=None)
        original_data = np.arange(40).reshape(20, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals[0].signal.shape, (20, 2))
        np.testing.assert_array_equal(f_signals[0].signal, original_data)

    def test_none_length_with_int_offset(self):
        """length=None with int offset should return from offset to end."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=5, length=None)
        original_data = np.arange(40).reshape(20, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals[0].signal.shape, (15, 2))
        np.testing.assert_array_equal(f_signals[0].signal, original_data[5:, :])

    def test_none_length_with_float_offset(self):
        """length=None with float offset should return from offset to end."""
        # offset=0.25 of 20 → 5
        sig_filter = RawSignalsFilterWindowSingleIF(offset=0.25, length=None)
        original_data = np.arange(40).reshape(20, 2).astype(np.float64)
        signals = RawSignals()
        signals.append(RawSignal(signal=original_data.copy()))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals[0].signal.shape, (15, 2))
        np.testing.assert_array_equal(f_signals[0].signal, original_data[5:, :])

    def test_none_length_multiple_signals(self):
        """length=None should work with multiple signals of different lengths."""
        sig_filter = RawSignalsFilterWindowSingleIF(offset=5, length=None)
        signals = RawSignals()
        signals.append(RawSignal(signal=np.ones((20, 2)) * 1.0))
        signals.append(RawSignal(signal=np.ones((30, 2)) * 2.0))

        f_signals = sig_filter.fit_transform(signals)
        self.assertEqual(f_signals[0].signal.shape, (15, 2))
        self.assertEqual(f_signals[1].signal.shape, (25, 2))


if __name__ == "__main__":
    unittest.main()
