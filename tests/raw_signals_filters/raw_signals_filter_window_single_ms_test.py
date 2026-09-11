
import unittest
import numpy as np
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_single_ms import RawSignalsFilterWindowSingleMs
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterWindowSingleMsBaseTest(RawSignalsFilterTest):
    """Inherited base filter tests."""
    __test__ = True

    def get_filters(self):
        # 0ms offset, 15ms length at 1000Hz = 15 samples
        return [RawSignalsFilterWindowSingleMs(offset_ms=0, length_ms=15)]


class RawSignalsFilterWindowSingleMsTest(unittest.TestCase):

    def _make_signals(self, n=3, rows=100, cols=4, sample_rate=1000):
        signals = RawSignals(sample_rate=sample_rate)
        for i in range(n):
            sig = np.arange(rows * cols).reshape(rows, cols).astype(np.float32) + i * 100
            signals.append(RawSignal(signal=sig, object_class=i, timestamp=i * 10, sample_rate=sample_rate))
        return signals

    def test_window_cut_ms(self):
        # 1000 Hz, 100 samples = 100ms total
        # offset=10ms, length=50ms -> samples 10..60 -> 50 samples
        signals = self._make_signals(n=3, rows=100, cols=4, sample_rate=1000)
        filt = RawSignalsFilterWindowSingleMs(offset_ms=10, length_ms=50)
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 3)
        for sig in result:
            self.assertEqual(len(sig), 50)

    def test_window_from_start(self):
        # offset=0ms, length=20ms at 1000Hz -> 20 samples
        signals = self._make_signals(n=2, rows=100, cols=4, sample_rate=1000)
        filt = RawSignalsFilterWindowSingleMs(offset_ms=0, length_ms=20)
        result = filt.fit_transform(signals)
        for sig in result:
            self.assertEqual(len(sig), 20)

    def test_different_sample_rate(self):
        # 2000 Hz, offset=10ms, length=50ms -> start=20, end=120 -> 100 samples
        signals = self._make_signals(n=2, rows=200, cols=4, sample_rate=2000)
        filt = RawSignalsFilterWindowSingleMs(offset_ms=10, length_ms=50)
        result = filt.fit_transform(signals)
        for sig in result:
            self.assertEqual(len(sig), 100)

    def test_preserves_metadata(self):
        signals = self._make_signals(n=2, rows=100, cols=4, sample_rate=1000)
        filt = RawSignalsFilterWindowSingleMs(offset_ms=0, length_ms=30)
        result = filt.fit_transform(signals)
        for i, sig in enumerate(result):
            self.assertEqual(sig.object_class, i)
            self.assertEqual(sig.timestamp, i * 10)

    def test_preserves_channel_names(self):
        signals = RawSignals(sample_rate=1000)
        sig = np.random.random((100, 3)).astype(np.float32)
        signals.append(RawSignal(signal=sig, channel_names=["A", "B", "C"], sample_rate=1000))
        filt = RawSignalsFilterWindowSingleMs(offset_ms=0, length_ms=50)
        result = filt.fit_transform(signals)
        self.assertEqual(result[0].channel_names, ("A", "B", "C"))

    def test_preserves_values(self):
        signals = RawSignals(sample_rate=1000)
        sig = np.arange(200).reshape(100, 2).astype(np.float32)
        signals.append(RawSignal(signal=sig, sample_rate=1000))
        filt = RawSignalsFilterWindowSingleMs(offset_ms=10, length_ms=20)
        result = filt.fit_transform(signals)
        np.testing.assert_array_equal(result[0].signal, sig[10:30])

    def test_not_fitted(self):
        signals = self._make_signals()
        filt = RawSignalsFilterWindowSingleMs(offset_ms=0, length_ms=10)
        with self.assertRaises(NotFittedError):
            filt.transform(signals)

    def test_number_of_signals_unchanged(self):
        signals = self._make_signals(n=5, rows=100, cols=4, sample_rate=1000)
        filt = RawSignalsFilterWindowSingleMs(offset_ms=0, length_ms=50)
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 5)

    def test_500hz_sample_rate(self):
        # 500 Hz, offset=20ms, length=100ms -> start=10, end=60 -> 50 samples
        signals = self._make_signals(n=2, rows=200, cols=4, sample_rate=500)
        filt = RawSignalsFilterWindowSingleMs(offset_ms=20, length_ms=100)
        result = filt.fit_transform(signals)
        for sig in result:
            self.assertEqual(len(sig), 50)


if __name__ == '__main__':
    unittest.main()
