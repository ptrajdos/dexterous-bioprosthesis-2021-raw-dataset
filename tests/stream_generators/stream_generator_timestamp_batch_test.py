import unittest

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.istream_generator import (
    IStreamGenerator,
)
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.stream_generator_timestamp_batch import (
    StreamGeneratorTimestampBatch,
)


def make_signals(timestamps, labels=None):
    signals = RawSignals(sample_rate=1000)
    labels = labels if labels is not None else [i % 3 for i in range(len(timestamps))]
    for ts, lab in zip(timestamps, labels):
        signals.append(
            RawSignal(np.random.randn(20, 2), object_class=lab, timestamp=ts)
        )
    return signals


class StreamGeneratorTimestampBatchTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.timestamps = [5.0, 1.0, 3.0, 2.0, 4.0, 0.0, 6.0]
        self.signals = make_signals(self.timestamps)

    def collect(self, stream):
        chunks = []
        while not stream.is_dry():
            chunks.append(stream.get_chunk())
        return chunks

    def test_is_interface(self):
        self.assertIsInstance(
            StreamGeneratorTimestampBatch(self.signals), IStreamGenerator
        )

    def test_default_one_sample_per_batch(self):
        stream = StreamGeneratorTimestampBatch(self.signals)
        self.assertEqual(stream.n_chunks, len(self.signals))
        chunks = self.collect(stream)
        self.assertTrue(all(len(X) == 1 for X, _ in chunks))
        ts = [X.get_timestamps()[0] for X, _ in chunks]
        self.assertEqual(ts, sorted(self.timestamps))

    def test_batch_size(self):
        stream = StreamGeneratorTimestampBatch(self.signals, batch_size=3)
        self.assertEqual(stream.n_chunks, 3)
        chunks = self.collect(stream)
        self.assertEqual([len(X) for X, _ in chunks], [3, 3, 1])
        all_ts = np.concatenate([X.get_timestamps() for X, _ in chunks])
        np.testing.assert_array_equal(all_ts, sorted(self.timestamps))

    def test_n_batches(self):
        stream = StreamGeneratorTimestampBatch(self.signals, n_batches=3)
        self.assertEqual(stream.n_chunks, 3)
        chunks = self.collect(stream)
        self.assertEqual([len(X) for X, _ in chunks], [3, 2, 2])
        all_ts = np.concatenate([X.get_timestamps() for X, _ in chunks])
        np.testing.assert_array_equal(all_ts, sorted(self.timestamps))

    def test_labels_match_signals(self):
        stream = StreamGeneratorTimestampBatch(self.signals, batch_size=2)
        for X, y in self.collect(stream):
            np.testing.assert_array_equal(X.get_labels(), y)

    def test_classes(self):
        stream = StreamGeneratorTimestampBatch(self.signals)
        np.testing.assert_array_equal(stream.classes_, [0, 1, 2])

    def test_stable_order_for_equal_timestamps(self):
        signals = make_signals([1.0, 0.0, 1.0, 1.0], labels=["a", "b", "c", "d"])
        stream = StreamGeneratorTimestampBatch(signals, batch_size=4)
        _, y = stream.get_chunk()
        np.testing.assert_array_equal(y, ["b", "a", "c", "d"])

    def test_chunk_bookkeeping(self):
        # Attributes used by strlearn evaluators (e.g. TestThenTrain).
        stream = StreamGeneratorTimestampBatch(self.signals, batch_size=3)
        self.assertEqual(stream.chunk_id, -1)
        self.assertIsNone(stream.previous_chunk)
        first = stream.get_chunk()
        self.assertEqual(stream.chunk_id, 0)
        self.assertIsNone(stream.previous_chunk)
        stream.get_chunk()
        self.assertEqual(stream.chunk_id, 1)
        self.assertIs(stream.previous_chunk, first)
        stream.get_chunk()
        self.assertTrue(stream.is_dry())
        with self.assertRaises(StopIteration):
            stream.get_chunk()

    def test_iteration_and_reset(self):
        stream = StreamGeneratorTimestampBatch(self.signals, batch_size=2)
        n = len(list(stream))
        self.assertEqual(n, stream.n_chunks)
        self.assertTrue(stream.is_dry())
        stream.reset()
        self.assertFalse(stream.is_dry())
        self.assertEqual(len(list(stream)), n)

    def test_test_then_train_like_loop(self):
        # Mimics strlearn.evaluators.TestThenTrain.process
        stream = StreamGeneratorTimestampBatch(self.signals, n_batches=4)
        tested = 0
        while True:
            X, y = stream.get_chunk()
            if stream.previous_chunk is not None:
                tested += 1
                self.assertLess(stream.chunk_id - 1, stream.n_chunks - 1)
            if stream.is_dry():
                break
        self.assertEqual(tested, stream.n_chunks - 1)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            StreamGeneratorTimestampBatch(self.signals, n_batches=2, batch_size=2)
        with self.assertRaises(ValueError):
            StreamGeneratorTimestampBatch(self.signals, batch_size=0)
        with self.assertRaises(ValueError):
            StreamGeneratorTimestampBatch(self.signals, n_batches=0)
        with self.assertRaises(ValueError):
            StreamGeneratorTimestampBatch(self.signals, n_batches=100)
        with self.assertRaises(ValueError):
            StreamGeneratorTimestampBatch(self.signals, batch_size=1.5)
        with self.assertRaises(ValueError):
            StreamGeneratorTimestampBatch(RawSignals())


if __name__ == "__main__":
    unittest.main()
