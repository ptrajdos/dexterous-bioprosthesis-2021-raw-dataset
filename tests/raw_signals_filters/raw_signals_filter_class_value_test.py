
import unittest
import numpy as np
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_class_value import RawSignalsFilterClassValue
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterClassValueBaseTest(RawSignalsFilterTest):
    """Inherited base filter tests."""
    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterClassValue(allowed_values=[0, 1, 2])]


class RawSignalsFilterClassValueTest(unittest.TestCase):

    def test_select_subset(self):
        signals = RawSignals()
        for i in range(5):
            sig = np.random.random((10, 3)).astype(np.float32)
            signals.append(RawSignal(signal=sig, object_class=i))

        filt = RawSignalsFilterClassValue(allowed_values=[1, 3])
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 2)
        labels = [s.object_class for s in result]
        self.assertEqual(labels, [1, 3])

    def test_no_match(self):
        signals = RawSignals()
        for i in range(5):
            sig = np.random.random((10, 3)).astype(np.float32)
            signals.append(RawSignal(signal=sig, object_class=i))

        filt = RawSignalsFilterClassValue(allowed_values=[10, 20])
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 0)

    def test_all_match(self):
        signals = RawSignals()
        for i in range(5):
            sig = np.random.random((10, 3)).astype(np.float32)
            signals.append(RawSignal(signal=sig, object_class=i))

        filt = RawSignalsFilterClassValue(allowed_values=[0, 1, 2, 3, 4])
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 5)

    def test_float_classes(self):
        signals = RawSignals()
        for v in [0.5, 1.5, 2.5, 3.5]:
            sig = np.random.random((10, 3)).astype(np.float32)
            signals.append(RawSignal(signal=sig, object_class=v))

        filt = RawSignalsFilterClassValue(allowed_values=[0.5, 2.5])
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 2)

    def test_string_classes(self):
        signals = RawSignals()
        for cls in ["a", "b", "c", "d"]:
            sig = np.random.random((10, 3)).astype(np.float32)
            signals.append(RawSignal(signal=sig, object_class=cls))

        filt = RawSignalsFilterClassValue(allowed_values=["a", "c"])
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].object_class, "a")
        self.assertEqual(result[1].object_class, "c")

    def test_numpy_scalar_classes(self):
        signals = RawSignals()
        for i in range(5):
            sig = np.random.random((10, 3)).astype(np.float32)
            signals.append(RawSignal(signal=sig, object_class=np.int32(i)))

        filt = RawSignalsFilterClassValue(allowed_values=[np.int32(0), np.int32(2), np.int32(4)])
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 3)

    def test_not_fitted(self):
        signals = RawSignals()
        sig = np.random.random((10, 3)).astype(np.float32)
        signals.append(RawSignal(signal=sig, object_class=0))

        filt = RawSignalsFilterClassValue(allowed_values=[0])
        with self.assertRaises(NotFittedError):
            filt.transform(signals)

    def test_preserves_signal_data(self):
        signals = RawSignals()
        sig = np.ones((10, 3)).astype(np.float32)
        signals.append(RawSignal(signal=sig, object_class=1, timestamp=42))
        signals.append(RawSignal(signal=sig * 2, object_class=2, timestamp=99))

        filt = RawSignalsFilterClassValue(allowed_values=[1])
        result = filt.fit_transform(signals)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0].signal, sig)
        self.assertEqual(result[0].timestamp, 42)


if __name__ == '__main__':
    unittest.main()
