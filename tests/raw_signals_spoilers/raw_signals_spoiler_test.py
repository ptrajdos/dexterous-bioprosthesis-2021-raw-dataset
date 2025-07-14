import unittest

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from tests.raw_signals_spoilers.raw_signals_spoiler_interface_test import (
    RawSignalsSpoilerInterfaceTest,
)


class RawSignalsSpoilerTest(RawSignalsSpoilerInterfaceTest):

    def get_spoiler_class(self):
        raise unittest.SkipTest("Skipping")

    def is_test_snr(self):
        return False

    def get_snrs(self):
        snrs = [-5, -1, 0, 1, 5, 10]
        return snrs

    def test_snrs(self):
        if self.is_test_snr():
            data = self.generate_sample_data()
            snrs = self.get_snrs()

            for snr in snrs:
                spoiler = self.get_spoiler_class()(snr=snr, channels_spoiled_frac=1.0)
                t_data = spoiler.fit_transform(data)

                self.assertEqual(
                    len(data), len(t_data), "Number of objects is different"
                )
                self.assertEqual(
                    data[0].to_numpy().shape,
                    t_data[0].to_numpy().shape,
                    "Different shapes",
                )

                for w_signal, o_signal in zip(t_data, data):
                    w_signal_np = w_signal.to_numpy()
                    o_signal_np = o_signal.to_numpy()

                    n_signal_np = w_signal_np - o_signal_np

                    c_snrs = spoiler._calculate_snrs(o_signal_np, n_signal_np)
                    d_snrs = np.zeros_like(c_snrs)
                    d_snrs[:] = snr

                    self.assertTrue(
                        np.allclose(c_snrs, d_snrs, rtol=1e-3, atol=1e-3),
                        "Too far from desired SNRs",
                    )

    def test_dtype(self):
        dtypes = [np.float32, np.float64, np.single, np.double]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                # Generate sample data with the specified dtype
                data = self.generate_sample_data(
                    samples_number=10, column_number=3, dtype=dtype
                )
                spoiler = self.get_spoiler_class()(snr=0, channels_spoiled_frac=1.0)
                t_data = spoiler.fit_transform(data)

                for w_signal in t_data:
                    np_sig = w_signal.to_numpy()
                    self.assertTrue(
                        np.issubdtype(np_sig.dtype, dtype),
                        f"Wrong dtype. Expected: {dtype}, got: {np_sig.dtype}"
                    )
