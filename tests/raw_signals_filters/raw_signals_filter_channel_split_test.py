import unittest

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_channel_split import (
    RawSignalsFilterChannelSplit,
)
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterChannelSplitTest(RawSignalsFilterTest):

    __test__ = True

    def get_filters(self):
        return [RawSignalsFilterChannelSplit()]

    def test_split(self):
        n_samples = 30
        n_channels = 3
        data = np.random.random((n_samples, n_channels))
        names = ["A", "B", "C"]
        signals = RawSignals(sample_rate=500)
        signals.append(RawSignal(signal=data, object_class=7, channel_names=names,
                                 timestamp=123))

        f_signals = RawSignalsFilterChannelSplit().fit_transform(signals)

        self.assertEqual(len(f_signals), n_channels)
        for ch_idx, sig in enumerate(f_signals):
            self.assertEqual(sig.signal.shape, (n_samples, 1))
            self.assertTrue(np.array_equal(sig.signal[:, 0], data[:, ch_idx]))
            self.assertEqual(tuple(sig.channel_names), (names[ch_idx],))
            self.assertEqual(sig.object_class, 7)
            self.assertEqual(sig.timestamp, 123)
            self.assertEqual(sig.sample_rate, 500)

    def test_split_count(self):
        signals = self.generate_sample_data(samples_number=10)
        f_signals = RawSignalsFilterChannelSplit().fit_transform(signals)
        expected = sum(s.signal.shape[1] for s in signals)
        self.assertEqual(len(f_signals), expected)
        for sig in f_signals:
            self.assertEqual(sig.signal.shape[1], 1)

    def test_split_copies_data(self):
        data = np.random.random((10, 2))
        signals = RawSignals()
        signals.append(RawSignal(signal=data))
        f_signals = RawSignalsFilterChannelSplit().fit_transform(signals)
        f_signals[0].signal[:] = -1
        self.assertFalse(np.any(data == -1))


if __name__ == "__main__":
    unittest.main()
