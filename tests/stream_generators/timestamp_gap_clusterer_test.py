import unittest

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.timestamp_gap_clusterer import (
    TimestampGapClusterer,
)


def sessions(starts, n_per=50, spacing=1.0, seed=0):
    rng = np.random.RandomState(seed)
    ts = np.concatenate(
        [s + np.cumsum(rng.uniform(0.5, 1.5, n_per)) * spacing for s in starts]
    )
    return ts


class TimestampGapClustererTest(unittest.TestCase):

    def test_detects_sessions(self):
        ts = sessions([0, 1000, 5000, 5300])
        cl = TimestampGapClusterer().fit(ts)
        self.assertEqual(cl.n_clusters_, 4)
        np.testing.assert_array_equal(cl.labels_, np.repeat([0, 1, 2, 3], 50))

    def test_unsorted_input_labels_time_ordered(self):
        ts = sessions([0, 1000, 5000])
        perm = np.random.RandomState(1).permutation(len(ts))
        labels = TimestampGapClusterer().fit_predict(ts[perm].reshape(-1, 1))
        expected = np.repeat([0, 1, 2], 50)[perm]
        np.testing.assert_array_equal(labels, expected)

    def test_single_cluster(self):
        ts = np.arange(100, dtype=float)
        cl = TimestampGapClusterer().fit(ts)
        self.assertEqual(cl.n_clusters_, 1)
        self.assertTrue(np.all(cl.labels_ == 0))

    def test_single_value_and_constant(self):
        self.assertEqual(TimestampGapClusterer().fit([3.0]).n_clusters_, 1)
        self.assertEqual(TimestampGapClusterer().fit(np.zeros(10)).n_clusters_, 1)

    def test_min_gap(self):
        ts = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 100], dtype=float)
        self.assertEqual(TimestampGapClusterer().fit(ts).n_clusters_, 3)
        self.assertEqual(TimestampGapClusterer(min_gap=5).fit(ts).n_clusters_, 2)

    def test_iqr_factor(self):
        ts = sessions([0, 200])  # moderate gap
        self.assertEqual(TimestampGapClusterer(iqr_factor=3).fit(ts).n_clusters_, 2)
        self.assertEqual(TimestampGapClusterer(iqr_factor=1e6).fit(ts).n_clusters_, 1)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            TimestampGapClusterer().fit([])
        with self.assertRaises(ValueError):
            TimestampGapClusterer().fit(np.zeros((5, 2)))
        with self.assertRaises(ValueError):
            TimestampGapClusterer(iqr_factor=-1).fit([1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
