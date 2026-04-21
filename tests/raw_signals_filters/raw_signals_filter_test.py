import abc
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsFilterTest(unittest.TestCase):

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise unittest.SkipTest("Skipping")

    @abc.abstractmethod
    def get_filters(self):
        """
        gets the filter to be tested
        """
        raise unittest.SkipTest("Skipping")

    def generate_sample_data(
        self,
        signal_number=10,
        column_number=3,
        samples_number=12,
        dtype=np.float32,
        labels=[0, 1, 2],
    ) -> RawSignals:
        signals = RawSignals()

        for i in range(1, signal_number + 1):
            label = np.random.choice(labels, 1)
            signals.append(
                RawSignal(
                    signal=np.random.random((samples_number, column_number)).astype(
                        dtype
                    ),
                    object_class=label,
                )
            )

        return signals

    def generate_zero_data(
        self, signal_number=10, column_number=3, samples_number=12, dtype=np.float32
    ) -> RawSignals:
        signals = RawSignals()

        for i in range(1, signal_number + 1):
            signals.append(
                RawSignal(
                    signal=np.zeros((samples_number, column_number)).astype(dtype)
                )
            )

        return signals

    def test_filter_fit_then_transform(self):
        filters = self.get_filters()

        for N, M, C in [(10, 30, 6), (2, 30, 3), (1, 30, 2), (5, 30, 10), (3, 50, 1)]:
            for filter in filters:
                signals = RawSignals()
                with self.subTest(filter=filter, N=N, M=M, C=C):

                    for i in range(1, N + 1):
                        signals.append(RawSignal(signal=np.random.random((M * i, C))))

                    try:
                        filter.fit(signals)
                        tr_signals = filter.transform(signals)

                        self.assertIsNotNone(tr_signals, "Transformed object is none.")
                        self.assertIsInstance(tr_signals, RawSignals)
                        self.assertTrue(
                            id(signals) != id(tr_signals), "Objects are the same"
                        )

                        for sig in tr_signals:
                            np_data = sig.to_numpy()
                            self.assertFalse(
                                np.any(np.isnan(np_data)), "NaN values in the data."
                            )
                            self.assertFalse(
                                np.any(np.isinf(np_data)), "Inf values in the data."
                            )

                    except Exception as ex:
                        self.fail("An exception has been caught: {}".format(ex))

    def test_filter_fit_then_transform_zeros(self):
        filters = self.get_filters()

        for N, M, C in [(10, 30, 6), (2, 30, 3), (1, 30, 2), (5, 30, 10), (3, 50, 1)]:
            for filter in filters:
                signals = RawSignals()
                with self.subTest(filter=filter, N=N, M=M, C=C):

                    for i in range(1, N + 1):
                        signals.append(RawSignal(signal=np.zeros((M * i, C))))

                    try:
                        filter.fit(signals)
                        tr_signals = filter.transform(signals)

                        self.assertIsNotNone(tr_signals, "Transformed object is none.")
                        self.assertIsInstance(tr_signals, RawSignals)
                        self.assertTrue(
                            id(signals) != id(tr_signals), "Objects are the same"
                        )

                        for sig in tr_signals:
                            np_data = sig.to_numpy()
                            self.assertFalse(
                                np.any(np.isnan(np_data)), "NaN values in the data."
                            )
                            self.assertFalse(
                                np.any(np.isinf(np_data)), "Inf values in the data."
                            )

                    except Exception as ex:
                        self.fail("An exception has been caught: {}".format(ex))

    def test_fit_transform(self):
        filters = self.get_filters()
        signals = self.generate_sample_data(samples_number=30)
        for filter in filters:
            try:
                tr_signals = filter.fit_transform(signals)

                self.assertIsNotNone(tr_signals, "Transformed object is none.")
                self.assertIsInstance(tr_signals, RawSignals)
                self.assertTrue(id(signals) != id(tr_signals), "Objects are the same")

            except Exception as ex:
                self.fail("An exception has been caught: {}".format(ex))

    def test_dtype(self):
        filters = self.get_filters()
        dtypes = [np.float32, np.float64, np.single, np.double]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                signals = self.generate_sample_data(
                    samples_number=30, column_number=3, dtype=dtype
                )

                for filter in filters:
                    try:
                        tr_signals = filter.fit_transform(signals)

                        for sig in tr_signals:
                            self.assertTrue(
                                np.issubdtype(sig.to_numpy().dtype, dtype),
                                f"Wrong dtype. Expected: {dtype}, got: {sig.to_numpy().dtype}",
                            )

                    except Exception as ex:
                        self.fail("An exception has been caught: {}".format(ex))

    def test_ydtype(self):
        filters = self.get_filters()
        dtypes = [np.int32, np.int64, np.float32, np.str_]
        classes = np.asanyarray([1, 2, 3])
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                signals = self.generate_sample_data(
                    samples_number=30, column_number=3, labels=classes.astype(dtype)
                )

                for filter in filters:

                    tr_signals: RawSignals = filter.fit_transform(signals)
                    labels = tr_signals.get_labels()
                    self.assertIsNotNone(labels, "Labels are none!")
                    self.assertIsInstance(labels, np.ndarray, "Wrong labels array type")
                    self.assertTrue(np.can_cast(labels.dtype, dtype), "Cannot cast")

                    if dtype != np.str_:
                        self.assertTrue(
                            labels.dtype == dtype,
                            f"Wrong exact type. Got {labels.dtype} expect: {dtype} ",
                        )

    def test_not_fitted(self):
        filters = self.get_filters()
        signals = self.generate_sample_data(samples_number=30)
        for filter in filters:
            with self.subTest(filter=filter):
                with self.assertRaises(NotFittedError):
                    filter.transform(signals)


if __name__ == "__main__":
    unittest.main()
