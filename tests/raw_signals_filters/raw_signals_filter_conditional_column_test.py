
import unittest
import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_conditional_column import RawSignalsFilterConditionalColumn
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_column_standarizer import RawSignalsFilterColumnStandarizer
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterConditionalColumnTest(RawSignalsFilterTest):
    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterConditionalColumn(("C", RawSignalsFilterColumnStandarizer()))]

    def test_only_matching_columns_transformed(self):
        signals = RawSignals()
        signal = np.array([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0]], dtype=np.float32)
        channel_names = ["EMG_1", "EMG_2", "ACC_X", "ACC_Y"]
        signals.append(RawSignal(signal=signal, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterConditionalColumn(("^EMG", RawSignalsFilterColumnStandarizer()))
        result = filt.fit_transform(signals)

        self.assertEqual(result[0].signal.shape, (3, 4))
        self.assertEqual(list(result[0].channel_names), ["EMG_1", "EMG_2", "ACC_X", "ACC_Y"])
        # Non-matching columns should be unchanged
        np.testing.assert_array_equal(result[0].signal[:, 2], signal[:, 2])
        np.testing.assert_array_equal(result[0].signal[:, 3], signal[:, 3])
        # Matching columns should be standardized (zero mean)
        np.testing.assert_allclose(result[0].signal[:, 0].mean(), 0.0, atol=1e-5)
        np.testing.assert_allclose(result[0].signal[:, 1].mean(), 0.0, atol=1e-5)

    def test_no_columns_match(self):
        signals = RawSignals()
        signal = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        channel_names = ["A", "B"]
        signals.append(RawSignal(signal=signal, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterConditionalColumn(("^Z", RawSignalsFilterColumnStandarizer()))
        result = filt.fit_transform(signals)

        np.testing.assert_array_equal(result[0].signal, signal)

    def test_all_columns_match(self):
        signals = RawSignals()
        signal = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        channel_names = ["EMG_1", "EMG_2"]
        signals.append(RawSignal(signal=signal, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterConditionalColumn(("^EMG", RawSignalsFilterColumnStandarizer()))
        result = filt.fit_transform(signals)

        np.testing.assert_allclose(result[0].signal[:, 0].mean(), 0.0, atol=1e-5)
        np.testing.assert_allclose(result[0].signal[:, 1].mean(), 0.0, atol=1e-5)

    def test_tuple_argument(self):
        filt = RawSignalsFilterConditionalColumn(("^EMG", RawSignalsFilterColumnStandarizer()))
        self.assertEqual(len(filt.column_filter_pairs), 1)
        self.assertEqual(filt.column_filter_pairs[0][0], "^EMG")

    def test_list_argument(self):
        filt = RawSignalsFilterConditionalColumn(["^EMG", RawSignalsFilterColumnStandarizer()])
        self.assertEqual(len(filt.column_filter_pairs), 1)
        self.assertEqual(filt.column_filter_pairs[0][0], "^EMG")

    def test_multiple_pairs(self):
        signals = RawSignals()
        signal = np.array([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0]], dtype=np.float32)
        channel_names = ["EMG_1", "EMG_2", "ACC_X", "ACC_Y"]
        signals.append(RawSignal(signal=signal, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterConditionalColumn([
            ("^EMG", RawSignalsFilterColumnStandarizer()),
            ("^ACC", RawSignalsFilterColumnStandarizer()),
        ])
        result = filt.fit_transform(signals)

        self.assertEqual(result[0].signal.shape, (3, 4))
        self.assertEqual(list(result[0].channel_names), channel_names)
        # All columns should be standardized (zero mean)
        for col in range(4):
            np.testing.assert_allclose(result[0].signal[:, col].mean(), 0.0, atol=1e-5)

    def test_multiple_pairs_no_columns_match(self):
        """No columns match any regex — all columns remain unchanged."""
        signals = RawSignals()
        signal = np.array([[1.0, 2.0, 3.0],
                           [5.0, 6.0, 7.0],
                           [9.0, 10.0, 11.0]], dtype=np.float32)
        channel_names = ["OTHER_1", "OTHER_2", "OTHER_3"]
        signals.append(RawSignal(signal=signal, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterConditionalColumn([
            ("^EMG", RawSignalsFilterColumnStandarizer()),
            ("^ACC", RawSignalsFilterColumnStandarizer()),
        ])
        result = filt.fit_transform(signals)

        self.assertEqual(result[0].signal.shape, (3, 3))
        self.assertEqual(list(result[0].channel_names), channel_names)
        np.testing.assert_array_equal(result[0].signal, signal)

    def test_multiple_pairs_partial_overlap(self):
        """Only EMG columns are standardized, ACC columns remain unchanged."""
        signals = RawSignals()
        signal = np.array([[1.0, 2.0, 3.0],
                           [5.0, 6.0, 7.0],
                           [9.0, 10.0, 11.0]], dtype=np.float32)
        channel_names = ["EMG_1", "ACC_X", "OTHER"]
        signals.append(RawSignal(signal=signal, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterConditionalColumn([
            ("^EMG", RawSignalsFilterColumnStandarizer()),
            ("^ACC", RawSignalsFilterColumnStandarizer()),
        ])
        result = filt.fit_transform(signals)

        # EMG and ACC standardized, OTHER unchanged
        np.testing.assert_allclose(result[0].signal[:, 0].mean(), 0.0, atol=1e-5)
        np.testing.assert_allclose(result[0].signal[:, 1].mean(), 0.0, atol=1e-5)
        np.testing.assert_array_equal(result[0].signal[:, 2], signal[:, 2])


if __name__ == '__main__':
    unittest.main()
