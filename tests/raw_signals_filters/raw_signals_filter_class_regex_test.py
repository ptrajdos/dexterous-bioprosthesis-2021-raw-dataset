
import unittest
import numpy as np
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_class_regex import RawSignalsFilterClassRegex
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterClassRegexBaseTest(RawSignalsFilterTest):
    """Inherited base filter tests with string labels."""
    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterClassRegex(pattern=".*")]

    def generate_sample_data(self, signal_number=10, column_number=3,
                             samples_number=12, dtype=np.float32,
                             labels=[0, 1, 2]) -> RawSignals:
        signals = RawSignals()
        for i in range(1, signal_number + 1):
            label = np.random.choice(labels, 1)
            signals.append(
                RawSignal(
                    signal=np.random.random((samples_number, column_number)).astype(dtype),
                    object_class=label,
                )
            )
        return signals


class RawSignalsFilterClassRegexTest(unittest.TestCase):

    def _make_signals_with_string_classes(self):
        signals = RawSignals()
        classes = ["grasp_open", "grasp_close", "pinch_open", "pinch_close", "rest"]
        for i, cls in enumerate(classes):
            sig = np.random.random((10, 3)).astype(np.float32)
            signals.append(RawSignal(signal=sig, object_class=cls))
        return signals, classes

    def test_select_by_regex(self):
        signals, _ = self._make_signals_with_string_classes()
        filt = RawSignalsFilterClassRegex(pattern="^grasp")
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 2)
        for sig in result:
            self.assertTrue(sig.object_class.startswith("grasp"))

    def test_no_match(self):
        signals, _ = self._make_signals_with_string_classes()
        filt = RawSignalsFilterClassRegex(pattern="^xyz")
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 0)

    def test_all_match(self):
        signals, _ = self._make_signals_with_string_classes()
        filt = RawSignalsFilterClassRegex(pattern=".*")
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 5)

    def test_numeric_class_as_string(self):
        signals = RawSignals()
        for i in range(5):
            sig = np.random.random((10, 3)).astype(np.float32)
            signals.append(RawSignal(signal=sig, object_class=i))
        filt = RawSignalsFilterClassRegex(pattern="^[012]$")
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 3)

    def test_array_class_regex(self):
        signals = RawSignals()
        sig = np.random.random((10, 3)).astype(np.float32)
        signals.append(RawSignal(signal=sig, object_class=np.array(["cat", "dog"])))
        signals.append(RawSignal(signal=sig, object_class=np.array(["fish", "bird"])))

        filt = RawSignalsFilterClassRegex(pattern="cat")
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 1)

    def test_not_fitted(self):
        signals, _ = self._make_signals_with_string_classes()
        filt = RawSignalsFilterClassRegex(pattern=".*")
        with self.assertRaises(NotFittedError):
            filt.transform(signals)

    def test_preserves_signal_data(self):
        signals = RawSignals()
        sig = np.ones((10, 3)).astype(np.float32)
        signals.append(RawSignal(signal=sig, object_class="keep", timestamp=42))
        signals.append(RawSignal(signal=sig * 2, object_class="drop", timestamp=99))

        filt = RawSignalsFilterClassRegex(pattern="^keep$")
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0].signal, sig)
        self.assertEqual(result[0].timestamp, 42)


if __name__ == '__main__':
    unittest.main()
