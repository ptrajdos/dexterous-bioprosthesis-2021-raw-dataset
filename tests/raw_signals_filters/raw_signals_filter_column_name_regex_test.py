
import unittest
import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_column_name_regex import RawSignalsFilterColumnNameRegex
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterColumnNameRegexTest(RawSignalsFilterTest):
    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterColumnNameRegex(pattern="C")]

    def test_regex_select(self):
        signals = RawSignals()
        signal = np.random.random((20, 4)).astype(np.float32)
        channel_names = ["EMG_1", "EMG_2", "ACC_X", "ACC_Y"]
        signals.append(RawSignal(signal=signal, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterColumnNameRegex(pattern="^EMG")
        filtered = filt.fit_transform(signals)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].signal.shape[1], 2)
        self.assertEqual(list(filtered[0].channel_names), ["EMG_1", "EMG_2"])

    def test_regex_no_match(self):
        signals = RawSignals()
        signal = np.random.random((20, 3)).astype(np.float32)
        channel_names = ["A", "B", "C"]
        signals.append(RawSignal(signal=signal, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterColumnNameRegex(pattern="^Z")
        filtered = filt.fit_transform(signals)

        self.assertEqual(filtered[0].signal.shape[1], 0)

    def test_regex_all_match(self):
        signals = RawSignals()
        signal = np.random.random((20, 3)).astype(np.float32)
        channel_names = ["EMG_1", "EMG_2", "EMG_3"]
        signals.append(RawSignal(signal=signal, object_class=0, channel_names=channel_names))

        filt = RawSignalsFilterColumnNameRegex(pattern="EMG")
        filtered = filt.fit_transform(signals)

        self.assertEqual(filtered[0].signal.shape[1], 3)


if __name__ == '__main__':
    unittest.main()
