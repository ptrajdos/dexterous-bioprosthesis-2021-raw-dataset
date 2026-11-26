"""Timestamp handling of filters that cut or split signals.

A filtered signal's timestamp must point at the time of its first sample,
i.e. original timestamp + (index of first kept sample) / sample_rate.
"""
import unittest

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_row_idx import (
    RawSignalsFilterRowIdx,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_row_range import (
    RawSignalsFilterRowRange,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_row_slice import (
    RawSignalsFilterRowSlice,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_segmentation import (
    RawSignalsFilterWindowSegmentation,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_segmentation_fs import (
    RawSignalsFilterWindowSegmentationFS,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_single import (
    RawSignalsFilterWindowSingle,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_single_if import (
    RawSignalsFilterWindowSingleIF,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_single_ms import (
    RawSignalsFilterWindowSingleMs,
)

FS = 2000
BASE_TS = 1_600_000_000.0
N_SAMPLES = 100


def make_signals(n_signals=3):
    signals = RawSignals(sample_rate=FS)
    for i in range(n_signals):
        signals.append(
            RawSignal(
                np.random.randn(N_SAMPLES, 2),
                object_class=0,
                timestamp=BASE_TS + i,
                sample_rate=FS,
            )
        )
    return signals


class RawSignalsFilterTimestampTest(unittest.TestCase):

    def check_single_output(self, sig_filter, first_sample):
        """Filters producing exactly one output signal per input signal."""
        signals = make_signals()
        out = sig_filter.fit_transform(signals)
        self.assertEqual(len(out), len(signals))
        for orig, new in zip(signals, out):
            self.assertAlmostEqual(new.timestamp, orig.timestamp + first_sample / FS)
            # Input must not be modified.
        for i, orig in enumerate(signals):
            self.assertEqual(orig.timestamp, BASE_TS + i)

    def check_segmentation(self, sig_filter, step):
        signals = make_signals()
        out = sig_filter.fit_transform(signals)
        per_signal = len(out) // len(signals)
        self.assertGreater(per_signal, 1)
        for k, new in enumerate(out):
            sig_idx, win_idx = divmod(k, per_signal)
            expected = BASE_TS + sig_idx + win_idx * step / FS
            self.assertAlmostEqual(new.timestamp, expected)
        for i, orig in enumerate(signals):
            self.assertEqual(orig.timestamp, BASE_TS + i)

    def test_window_segmentation_samples(self):
        self.check_segmentation(
            RawSignalsFilterWindowSegmentation(window_length=20, overlap=5), step=15
        )

    def test_window_segmentation_fraction(self):
        # window = round(0.2*100) = 20, overlap = round(0.25*20) = 5
        self.check_segmentation(
            RawSignalsFilterWindowSegmentation(window_length=0.2, overlap=0.25),
            step=15,
        )

    def test_window_segmentation_fs(self):
        # 10 ms @ 2 kHz = 20 samples, 2.5 ms = 5 samples
        self.check_segmentation(
            RawSignalsFilterWindowSegmentationFS(window_length=10, overlap=2.5),
            step=15,
        )

    def test_window_single(self):
        self.check_single_output(
            RawSignalsFilterWindowSingle(start_sample=10, end_sample=30), 10
        )

    def test_window_single_negative_start(self):
        self.check_single_output(
            RawSignalsFilterWindowSingle(start_sample=-20, end_sample=None), 80
        )

    def test_window_single_if_int(self):
        self.check_single_output(RawSignalsFilterWindowSingleIF(offset=7, length=10), 7)

    def test_window_single_if_float(self):
        self.check_single_output(
            RawSignalsFilterWindowSingleIF(offset=0.25, length=None), 25
        )

    def test_window_single_ms(self):
        # 5 ms @ 2 kHz = 10 samples
        self.check_single_output(
            RawSignalsFilterWindowSingleMs(offset_ms=5, length_ms=10), 10
        )

    def test_row_range(self):
        self.check_single_output(RawSignalsFilterRowRange(start=12, stop=40, step=2), 12)

    def test_row_range_none_start(self):
        self.check_single_output(RawSignalsFilterRowRange(stop=40), 0)

    def test_row_slice(self):
        self.check_single_output(RawSignalsFilterRowSlice(row_slice=slice(33, 50)), 33)

    def test_row_idx(self):
        self.check_single_output(RawSignalsFilterRowIdx(indices_list=[4, 5, 9]), 4)


if __name__ == "__main__":
    unittest.main()
