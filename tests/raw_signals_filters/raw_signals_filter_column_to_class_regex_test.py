
import unittest
import numpy as np
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_column_to_class_regex import RawSignalsFilterColumnToClassRegex


class RawSignalsFilterColumnToClassRegexTest(unittest.TestCase):

    def test_basic_extraction(self):
        signals = RawSignals()
        sig = np.arange(20).reshape(4, 5).astype(np.float32)
        channel_names = ["EMG_1", "EMG_2", "ACC_X", "ACC_Y", "GYRO"]
        signals.append(RawSignal(signal=sig, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterColumnToClassRegex(pattern="^ACC")
        result = filt.fit_transform(signals)

        self.assertEqual(result[0].signal.shape[1], 3, "Remaining signal should have 3 columns")
        self.assertIsInstance(result[0].object_class, np.ndarray)
        self.assertEqual(result[0].object_class.shape, (4, 2))
        np.testing.assert_array_equal(result[0].object_class, sig[:, [2, 3]])
        self.assertEqual(list(result[0].channel_names), ["EMG_1", "EMG_2", "GYRO"])

    def test_no_match(self):
        signals = RawSignals()
        sig = np.random.random((10, 3)).astype(np.float32)
        channel_names = ["A", "B", "C"]
        signals.append(RawSignal(signal=sig, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterColumnToClassRegex(pattern="^Z")
        result = filt.fit_transform(signals)

        self.assertEqual(result[0].signal.shape[1], 3)
        self.assertEqual(result[0].object_class.shape, (10, 0))

    def test_all_match(self):
        signals = RawSignals()
        sig = np.random.random((10, 3)).astype(np.float32)
        channel_names = ["EMG_1", "EMG_2", "EMG_3"]
        signals.append(RawSignal(signal=sig, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterColumnToClassRegex(pattern="EMG")
        result = filt.fit_transform(signals)

        self.assertEqual(result[0].signal.shape[1], 0)
        self.assertEqual(result[0].object_class.shape, (10, 3))

    def test_preserves_timestamp(self):
        signals = RawSignals()
        sig = np.random.random((10, 3)).astype(np.float32)
        signals.append(RawSignal(signal=sig, object_class=0, channel_names=["A", "B", "C"], timestamp=99))

        filt = RawSignalsFilterColumnToClassRegex(pattern="^A")
        result = filt.fit_transform(signals)

        self.assertEqual(result[0].timestamp, 99)

    def test_not_fitted(self):
        signals = RawSignals()
        sig = np.random.random((10, 3)).astype(np.float32)
        signals.append(RawSignal(signal=sig, object_class=0))

        filt = RawSignalsFilterColumnToClassRegex(pattern="C")
        with self.assertRaises(NotFittedError):
            filt.transform(signals)

    def test_multiple_signals(self):
        signals = RawSignals()
        for i in range(5):
            sig = np.random.random((10, 4)).astype(np.float32)
            signals.append(RawSignal(signal=sig, object_class=i, channel_names=["X1", "Y1", "X2", "Y2"]))

        filt = RawSignalsFilterColumnToClassRegex(pattern="^Y")
        result = filt.fit_transform(signals)

        self.assertEqual(len(result), 5)
        for s in result:
            self.assertEqual(s.signal.shape[1], 2)
            self.assertEqual(s.object_class.shape, (10, 2))
            self.assertEqual(list(s.channel_names), ["X1", "X2"])


if __name__ == '__main__':
    unittest.main()
