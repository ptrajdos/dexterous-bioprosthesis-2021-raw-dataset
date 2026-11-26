import unittest

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_rms import (
    NpSignalExtractorRms,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import ASetCreator
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_dwt import (
    SetCreatorDWT,
)
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.istream_generator import (
    IStreamGenerator,
)
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.stream_generator_feature_extractor import (
    StreamGeneratorFeatureExtractor,
)
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.stream_generator_timestamp_batch import (
    StreamGeneratorTimestampBatch,
)


class RecordingSetCreator(ASetCreator):
    """Records what it was fitted on; feature = signal mean minus fitted mean."""

    def __init__(self) -> None:
        super().__init__()
        self.fit_calls = 0
        self.fitted_timestamps = None
        self.mean_ = None

    def fit(self, raw_signals, y=None):
        super().fit(raw_signals, y)
        self.fit_calls += 1
        self.fitted_timestamps = list(raw_signals.get_timestamps())
        self.mean_ = np.mean(raw_signals.to_numpy_concat())
        return self

    def transform(self, raw_signals):
        X = np.asarray([[np.mean(s.to_numpy()) - self.mean_] for s in raw_signals])
        return X, raw_signals.get_labels(), raw_signals.get_timestamps()

    def fit_transform(self, raw_signals, y=None):
        return self.fit(raw_signals).transform(raw_signals)

    def get_channel_attribs_indices(self):
        return None

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["mean_diff"])


def make_signals(n=8):
    signals = RawSignals(sample_rate=1000)
    for i in range(n):
        signals.append(
            RawSignal(
                np.random.randn(64, 2) + i,
                object_class=i % 2,
                timestamp=float(n - i),  # reverse order vs. insertion
            )
        )
    return signals


class StreamGeneratorFeatureExtractorTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.signals = make_signals()

    def collect(self, stream):
        chunks = []
        while not stream.is_dry():
            chunks.append(stream.get_chunk())
        return chunks

    def test_is_interface(self):
        stream = StreamGeneratorFeatureExtractor(
            StreamGeneratorTimestampBatch(self.signals), RecordingSetCreator()
        )
        self.assertIsInstance(stream, IStreamGenerator)

    def test_fits_only_on_first_chunk(self):
        creator = RecordingSetCreator()
        stream = StreamGeneratorFeatureExtractor(
            StreamGeneratorTimestampBatch(self.signals, batch_size=3), creator
        )
        self.collect(stream)
        self.assertEqual(stream.set_creator_.fit_calls, 1)
        # Earliest three timestamps.
        self.assertEqual(stream.set_creator_.fitted_timestamps, [1.0, 2.0, 3.0])
        # The original set creator is untouched.
        self.assertEqual(creator.fit_calls, 0)
        self.assertIsNone(creator.mean_)

    def test_chunks_are_numeric(self):
        stream = StreamGeneratorFeatureExtractor(
            StreamGeneratorTimestampBatch(self.signals, n_batches=3), RecordingSetCreator()
        )
        chunks = self.collect(stream)
        self.assertEqual(len(chunks), 3)
        self.assertEqual([X.shape for X, _ in chunks], [(3, 1), (3, 1), (2, 1)])
        for X, y in chunks:
            self.assertIsInstance(X, np.ndarray)
            self.assertEqual(len(X), len(y))

    def test_features_use_first_chunk_model(self):
        inner = StreamGeneratorTimestampBatch(self.signals, batch_size=4)
        stream = StreamGeneratorFeatureExtractor(inner, RecordingSetCreator())
        chunks = self.collect(stream)

        ordered = self.signals[list(np.argsort(self.signals.get_timestamps()))]
        first_mean = np.mean(ordered[:4].to_numpy_concat())
        expected = np.asarray([np.mean(s.to_numpy()) - first_mean for s in ordered])
        got = np.concatenate([X[:, 0] for X, _ in chunks])
        np.testing.assert_allclose(got, expected)

    def test_bookkeeping_delegates_to_inner_stream(self):
        inner = StreamGeneratorTimestampBatch(self.signals, batch_size=3)
        stream = StreamGeneratorFeatureExtractor(inner, RecordingSetCreator())
        self.assertEqual(stream.n_chunks, inner.n_chunks)
        np.testing.assert_array_equal(stream.classes_, inner.classes_)
        self.assertEqual(stream.chunk_id, -1)
        self.assertIsNone(stream.previous_chunk)
        first = stream.get_chunk()
        self.assertEqual(stream.chunk_id, 0)
        self.assertIsNone(stream.previous_chunk)
        stream.get_chunk()
        self.assertEqual(stream.chunk_id, 1)
        self.assertIs(stream.previous_chunk, first)

    def test_reset_refits(self):
        stream = StreamGeneratorFeatureExtractor(
            StreamGeneratorTimestampBatch(self.signals, batch_size=2), RecordingSetCreator()
        )
        first_pass = [X for X, _ in self.collect(stream)]
        fitted_first = stream.set_creator_
        stream.reset()
        self.assertIsNone(stream.set_creator_)
        self.assertEqual(stream.chunk_id, -1)
        second_pass = [X for X, _ in self.collect(stream)]
        self.assertIsNot(stream.set_creator_, fitted_first)
        for a, b in zip(first_pass, second_pass):
            np.testing.assert_allclose(a, b)

    def test_iteration(self):
        stream = StreamGeneratorFeatureExtractor(
            StreamGeneratorTimestampBatch(self.signals, batch_size=3), RecordingSetCreator()
        )
        self.assertEqual(len(list(stream)), stream.n_chunks)

    def test_with_real_set_creator(self):
        stream = StreamGeneratorFeatureExtractor(
            StreamGeneratorTimestampBatch(self.signals, batch_size=3),
            SetCreatorDWT(num_levels=1, extractors=[NpSignalExtractorRms()]),
        )
        chunks = self.collect(stream)
        n_features = chunks[0][0].shape[1]
        for X, y in chunks:
            self.assertEqual(X.ndim, 2)
            self.assertEqual(X.shape[1], n_features)
            self.assertTrue(np.all(np.isfinite(X)))

    def test_test_then_train_like_loop(self):
        # Mimics strlearn.evaluators.TestThenTrain.process with a partial_fit model.
        from sklearn.naive_bayes import GaussianNB

        stream = StreamGeneratorFeatureExtractor(
            StreamGeneratorTimestampBatch(self.signals, batch_size=2),
            SetCreatorDWT(num_levels=1, extractors=[NpSignalExtractorRms()]),
        )
        clf = GaussianNB()
        scores = np.zeros(stream.n_chunks - 1)
        while True:
            X, y = stream.get_chunk()
            if stream.previous_chunk is not None:
                scores[stream.chunk_id - 1] = np.mean(clf.predict(X) == y)
            clf.partial_fit(X, y, classes=stream.classes_)
            if stream.is_dry():
                break
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))


if __name__ == "__main__":
    unittest.main()
