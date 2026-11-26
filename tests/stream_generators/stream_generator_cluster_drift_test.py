import unittest

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.stream_generator_cluster_drift import (
    StreamGeneratorClusterDrift,
)
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.stream_generator_feature_extractor import (
    StreamGeneratorFeatureExtractor,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_rms import (
    NpSignalExtractorRms,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_dwt import SetCreatorDWT

N_PER = 40
STARTS = [0.0, 1000.0, 3000.0]


def make_signals(starts=STARTS, n_per=N_PER, shuffle=True, seed=0):
    rng = np.random.RandomState(seed)
    items = []
    for c, s in enumerate(starts):
        ts = s + np.cumsum(rng.uniform(0.5, 1.5, n_per))
        for t in ts:
            items.append((t, c))
    if shuffle:
        items = [items[i] for i in rng.permutation(len(items))]
    signals = RawSignals(sample_rate=1000)
    for t, c in items:
        # Label = true session, to check the stream order.
        signals.append(RawSignal(rng.randn(32, 2), object_class=c, timestamp=t))
    return signals


class StreamGeneratorClusterDriftTest(unittest.TestCase):

    def setUp(self):
        self.signals = make_signals()

    def stream_labels(self, stream):
        return np.concatenate([y for _, y in stream])

    def test_detects_clusters(self):
        stream = StreamGeneratorClusterDrift(self.signals, random_state=0)
        self.assertEqual(stream.n_clusters_, 3)
        np.testing.assert_array_equal(stream.cluster_labels_, self.signals.get_labels())

    def test_permutation_of_all_signals(self):
        for mixing in [0.0, 0.1, 0.25, 0.5]:
            stream = StreamGeneratorClusterDrift(self.signals, mixing=mixing, random_state=0)
            self.assertEqual(sorted(stream.order_.tolist()), list(range(len(self.signals))))

    def test_no_mixing_is_timestamp_order(self):
        stream = StreamGeneratorClusterDrift(self.signals, mixing=0.0, batch_size=7)
        ts = np.concatenate([X.get_timestamps() for X, _ in stream])
        np.testing.assert_array_equal(ts, np.sort(self.signals.get_timestamps()))

    def test_mixing_zone(self):
        mixing = 0.25
        n_mix = int(mixing * N_PER)
        stream = StreamGeneratorClusterDrift(self.signals, mixing=mixing, random_state=0)
        labels = self.stream_labels(stream)

        # Pure parts: first cluster start, middle of cluster 1, last cluster end.
        self.assertTrue(np.all(labels[: N_PER - n_mix] == 0))
        self.assertTrue(np.all(labels[N_PER + n_mix : 2 * N_PER - n_mix] == 1))
        self.assertTrue(np.all(labels[2 * N_PER + n_mix :] == 2))

        # Transition zone 0 -> 1 contains both clusters, and is actually mixed.
        zone = labels[N_PER - n_mix : N_PER + n_mix]
        self.assertEqual(set(zone.tolist()), {0, 1})
        self.assertFalse(np.all(zone[:n_mix] == 0) and np.all(zone[n_mix:] == 1))

    def test_order_within_cluster_preserved(self):
        stream = StreamGeneratorClusterDrift(self.signals, mixing=0.5, random_state=3)
        chunks = list(stream)
        ts_stream = np.concatenate([X.get_timestamps() for X, _ in chunks])
        labels = np.concatenate([y for _, y in chunks])
        for c in range(3):
            ts_c = ts_stream[labels == c]
            self.assertTrue(np.all(np.diff(ts_c) > 0))

    def test_gradual_share(self):
        # Averaged over many seeds, the share of the next cluster rises along the zone.
        signals = make_signals(starts=[0.0, 1000.0], n_per=200)
        n_mix = 100
        shares = np.zeros(2 * n_mix)
        for seed in range(30):
            stream = StreamGeneratorClusterDrift(signals, mixing=0.5, random_state=seed)
            labels = self.stream_labels(stream)
            shares += labels[200 - n_mix : 200 + n_mix]
        shares /= 30
        quarter = len(shares) // 4
        self.assertLess(shares[:quarter].mean(), 0.3)
        self.assertGreater(shares[-quarter:].mean(), 0.7)
        self.assertLess(shares[:quarter].mean(), shares[quarter : 2 * quarter].mean())

    def test_random_state_reproducible(self):
        a = StreamGeneratorClusterDrift(self.signals, mixing=0.3, random_state=42)
        b = StreamGeneratorClusterDrift(self.signals, mixing=0.3, random_state=42)
        np.testing.assert_array_equal(a.order_, b.order_)

    def test_reset_replays_same_stream(self):
        stream = StreamGeneratorClusterDrift(self.signals, batch_size=10, random_state=0)
        first = self.stream_labels(stream)
        stream.reset()
        np.testing.assert_array_equal(first, self.stream_labels(stream))

    def test_batching(self):
        stream = StreamGeneratorClusterDrift(self.signals, n_batches=7, random_state=0)
        self.assertEqual(stream.n_chunks, 7)
        stream = StreamGeneratorClusterDrift(self.signals, random_state=0)
        self.assertEqual(stream.n_chunks, len(self.signals))

    def test_custom_clusterer(self):
        km = KMeans(n_clusters=3, n_init=10, random_state=0)
        stream = StreamGeneratorClusterDrift(self.signals, clusterer=km, random_state=0)
        self.assertEqual(stream.n_clusters_, 3)
        np.testing.assert_array_equal(stream.cluster_labels_, self.signals.get_labels())
        self.assertFalse(hasattr(km, "labels_"))  # passed clusterer not modified

    def test_noise_labels_assigned(self):
        signals = make_signals()
        signals.append(RawSignal(np.zeros((32, 2)), object_class=1, timestamp=1500.0))
        stream = StreamGeneratorClusterDrift(
            signals, clusterer=DBSCAN(eps=10, min_samples=3), random_state=0
        )
        self.assertEqual(stream.n_clusters_, 3)
        self.assertNotIn(-1, stream.cluster_labels_)
        self.assertEqual(sorted(stream.order_.tolist()), list(range(len(signals))))

    def test_small_clusters(self):
        signals = make_signals(starts=[0.0, 1000.0, 2000.0], n_per=1)
        signals += make_signals(starts=[5000.0], n_per=30)
        stream = StreamGeneratorClusterDrift(signals, mixing=0.5, random_state=0)
        self.assertEqual(sorted(stream.order_.tolist()), list(range(len(signals))))

    def test_invalid_mixing(self):
        with self.assertRaises(ValueError):
            StreamGeneratorClusterDrift(self.signals, mixing=0.6)
        with self.assertRaises(ValueError):
            StreamGeneratorClusterDrift(self.signals, mixing=-0.1)

    def test_with_feature_extractor(self):
        stream = StreamGeneratorFeatureExtractor(
            StreamGeneratorClusterDrift(self.signals, batch_size=10, random_state=0),
            SetCreatorDWT(num_levels=1, extractors=[NpSignalExtractorRms()]),
        )
        chunks = list(stream)
        self.assertEqual(len(chunks), stream.n_chunks)
        self.assertTrue(all(X.ndim == 2 for X, _ in chunks))


if __name__ == "__main__":
    unittest.main()
