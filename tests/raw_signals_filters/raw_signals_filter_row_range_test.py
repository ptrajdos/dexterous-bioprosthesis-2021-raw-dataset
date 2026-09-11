
import unittest
import numpy as np
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_row_range import RawSignalsFilterRowRange
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterRowRangeBaseTest(RawSignalsFilterTest):
    """Inherited base filter tests."""
    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterRowRange(start=0, stop=5)]


class RawSignalsFilterRowRangeTest(unittest.TestCase):

    def _make_signals(self, n=3, rows=20, cols=4):
        signals = RawSignals()
        for i in range(n):
            sig = np.arange(rows * cols).reshape(rows, cols).astype(np.float32) + i * 100
            signals.append(RawSignal(signal=sig, object_class=i, timestamp=i * 10))
        return signals

    def test_select_range(self):
        signals = self._make_signals(n=3, rows=20, cols=4)
        filt = RawSignalsFilterRowRange(start=5, stop=15)
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 3)
        for sig in result:
            self.assertEqual(len(sig), 10)

    def test_select_from_start(self):
        signals = self._make_signals(n=2, rows=20, cols=4)
        filt = RawSignalsFilterRowRange(stop=5)
        result = filt.fit_transform(signals)
        for sig in result:
            self.assertEqual(len(sig), 5)

    def test_select_to_end(self):
        signals = self._make_signals(n=2, rows=20, cols=4)
        filt = RawSignalsFilterRowRange(start=15)
        result = filt.fit_transform(signals)
        for sig in result:
            self.assertEqual(len(sig), 5)

    def test_step(self):
        signals = self._make_signals(n=2, rows=20, cols=4)
        filt = RawSignalsFilterRowRange(start=0, stop=20, step=2)
        result = filt.fit_transform(signals)
        for sig in result:
            self.assertEqual(len(sig), 10)

    def test_preserves_metadata(self):
        signals = self._make_signals(n=2, rows=20, cols=4)
        filt = RawSignalsFilterRowRange(start=0, stop=10)
        result = filt.fit_transform(signals)
        for i, sig in enumerate(result):
            self.assertEqual(sig.object_class, i)
            self.assertEqual(sig.timestamp, i * 10)

    def test_preserves_channel_names(self):
        signals = RawSignals()
        sig = np.random.random((20, 3)).astype(np.float32)
        signals.append(RawSignal(signal=sig, channel_names=["A", "B", "C"]))
        filt = RawSignalsFilterRowRange(start=0, stop=5)
        result = filt.fit_transform(signals)
        self.assertEqual(result[0].channel_names, ("A", "B", "C"))

    def test_preserves_values(self):
        signals = RawSignals()
        sig = np.arange(40).reshape(20, 2).astype(np.float32)
        signals.append(RawSignal(signal=sig))
        filt = RawSignalsFilterRowRange(start=5, stop=10)
        result = filt.fit_transform(signals)
        np.testing.assert_array_equal(result[0].signal, sig[5:10])

    def test_not_fitted(self):
        signals = self._make_signals()
        filt = RawSignalsFilterRowRange(start=0, stop=5)
        with self.assertRaises(NotFittedError):
            filt.transform(signals)

    def test_number_of_signals_unchanged(self):
        signals = self._make_signals(n=5, rows=20, cols=4)
        filt = RawSignalsFilterRowRange(start=0, stop=10)
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 5)


if __name__ == '__main__':
    unittest.main()
