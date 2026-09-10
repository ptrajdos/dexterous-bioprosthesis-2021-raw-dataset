
import unittest
import numpy as np
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_column_to_class_idx import RawSignalsFilterColumnToClassIdx


class RawSignalsFilterColumnToClassIdxTest(unittest.TestCase):

    def _make_signals(self, n_signals=3, n_samples=20, n_cols=5):
        signals = RawSignals()
        for i in range(n_signals):
            sig = np.random.random((n_samples, n_cols)).astype(np.float32)
            channel_names = [f"Ch{j}" for j in range(n_cols)]
            signals.append(RawSignal(signal=sig, object_class=i, channel_names=channel_names))
        return signals

    def test_basic_extraction(self):
        signals = self._make_signals(n_signals=2, n_samples=10, n_cols=5)
        indices = [1, 3]
        filt = RawSignalsFilterColumnToClassIdx(indices_list=indices)
        result = filt.fit_transform(signals)

        self.assertEqual(len(result), 2)
        for sig in result:
            self.assertEqual(sig.signal.shape[1], 3, "Remaining signal should have 3 columns")
            self.assertIsInstance(sig.object_class, np.ndarray)
            self.assertEqual(sig.object_class.shape, (10, 2), "Class should be (samples, 2)")

    def test_channel_names_updated(self):
        signals = self._make_signals(n_signals=1, n_samples=10, n_cols=4)
        indices = [0, 2]
        filt = RawSignalsFilterColumnToClassIdx(indices_list=indices)
        result = filt.fit_transform(signals)

        self.assertEqual(list(result[0].channel_names), ["Ch1", "Ch3"])

    def test_single_column_to_class(self):
        signals = self._make_signals(n_signals=1, n_samples=15, n_cols=4)
        original_col = signals[0].signal[:, 2].copy()
        filt = RawSignalsFilterColumnToClassIdx(indices_list=[2])
        result = filt.fit_transform(signals)

        self.assertEqual(result[0].signal.shape[1], 3)
        self.assertEqual(result[0].object_class.shape, (15, 1))
        np.testing.assert_allclose(result[0].object_class[:, 0], original_col, rtol=1e-6)

    def test_class_values_match_original_columns(self):
        signals = RawSignals()
        sig = np.arange(20).reshape(4, 5).astype(np.float32)
        signals.append(RawSignal(signal=sig, object_class=0, channel_names=["A", "B", "C", "D", "E"]))

        filt = RawSignalsFilterColumnToClassIdx(indices_list=[1, 4])
        result = filt.fit_transform(signals)

        expected_class = sig[:, [1, 4]]
        expected_signal = sig[:, [0, 2, 3]]
        np.testing.assert_array_equal(result[0].object_class, expected_class)
        np.testing.assert_array_equal(result[0].signal, expected_signal)

    def test_preserves_timestamp(self):
        signals = RawSignals()
        sig = np.random.random((10, 3)).astype(np.float32)
        signals.append(RawSignal(signal=sig, object_class=0, timestamp=42))

        filt = RawSignalsFilterColumnToClassIdx(indices_list=[0])
        result = filt.fit_transform(signals)

        self.assertEqual(result[0].timestamp, 42)

    def test_not_fitted(self):
        signals = self._make_signals()
        filt = RawSignalsFilterColumnToClassIdx(indices_list=[0])
        with self.assertRaises(NotFittedError):
            filt.transform(signals)

    def test_multiple_signals(self):
        signals = self._make_signals(n_signals=5, n_samples=10, n_cols=6)
        filt = RawSignalsFilterColumnToClassIdx(indices_list=[0, 5])
        result = filt.fit_transform(signals)

        self.assertEqual(len(result), 5)
        for sig in result:
            self.assertEqual(sig.signal.shape[1], 4)
            self.assertEqual(sig.object_class.shape, (10, 2))


if __name__ == '__main__':
    unittest.main()
