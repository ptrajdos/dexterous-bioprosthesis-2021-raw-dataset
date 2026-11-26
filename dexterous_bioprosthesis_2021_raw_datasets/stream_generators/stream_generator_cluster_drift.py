"""Module implementing a cluster-based stream with gradual drift.

Signals are clustered by timestamp (1D), clusters are served one after
another in time order, and neighbouring clusters are mixed near their
boundary to simulate gradual concept drift.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.stream_generator_batch import (
    AStreamGeneratorBatch,
)
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.timestamp_gap_clusterer import (
    TimestampGapClusterer,
)


class StreamGeneratorClusterDrift(AStreamGeneratorBatch):
    """Stream of timestamp clusters with gradual drift between them.

    Procedure:

    1. Timestamps are clustered with ``clusterer`` (1D). Clusters are
       ordered by their median timestamp; signals inside a cluster are
       ordered by timestamp.
    2. For every pair of neighbouring clusters A -> B, the last
       ``mixing * len(A)`` signals of A and the first ``mixing * len(B)``
       signals of B form a transition zone. In the zone, the signals are
       interleaved at random; the probability of taking the next signal
       from B grows linearly from 0 to 1 across the zone. The order within
       A and within B is preserved.
    3. The resulting sequence is split into batches like in
       :class:`StreamGeneratorTimestampBatch`.

    ``mixing=0`` gives abrupt drift (no mixing). The maximum ``mixing=0.5``
    makes every signal of a middle cluster part of a transition zone.
    """

    def __init__(
        self,
        raw_signals: RawSignals,
        n_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
        clusterer: Optional[BaseEstimator] = None,
        mixing: float = 0.25,
        random_state=None,
    ) -> None:
        """Create the stream.

        Arguments:
        ---------
        raw_signals -- signals to be streamed
        n_batches -- number of batches (exclusive with ``batch_size``)
        batch_size -- number of signals per batch; if neither ``n_batches``
            nor ``batch_size`` is given, each batch contains one signal
        clusterer -- scikit-learn style clusterer applied to timestamps of
            shape (n_signals, 1), must provide ``fit_predict``; defaults to
            :class:`TimestampGapClusterer`. Noise labels (-1, e.g. from
            DBSCAN/HDBSCAN) are assigned to the cluster of the nearest
            non-noise timestamp. The passed object is cloned, not modified.
        mixing -- fraction of each cluster, at each side, that takes part
            in the transition to the neighbouring cluster, in [0, 0.5]
        random_state -- seed or RandomState used for mixing

        Attributes set after construction:
        cluster_labels_ -- cluster label of each signal (0 = earliest cluster)
        n_clusters_ -- number of detected clusters
        clusterer_ -- fitted clusterer

        """
        if not (0.0 <= mixing <= 0.5):
            raise ValueError(f"mixing must be within [0, 0.5]. Got {mixing}")
        self.clusterer = clusterer
        self.mixing = mixing
        self.random_state = random_state
        super().__init__(raw_signals, n_batches=n_batches, batch_size=batch_size)

    def _cluster(self, timestamps: np.ndarray) -> np.ndarray:
        """Return time-ordered cluster labels (0..n_clusters-1) per signal."""
        base = self.clusterer if self.clusterer is not None else TimestampGapClusterer()
        self.clusterer_ = clone(base)
        labels = np.asarray(
            self.clusterer_.fit_predict(timestamps.reshape(-1, 1).astype(float))
        )

        noise = labels == -1
        if noise.all():
            labels = np.zeros_like(labels)
        elif noise.any():
            valid = np.flatnonzero(~noise)
            for i in np.flatnonzero(noise):
                nearest = valid[np.argmin(np.abs(timestamps[valid] - timestamps[i]))]
                labels[i] = labels[nearest]

        # Relabel so that clusters are numbered by median timestamp.
        unique = np.unique(labels)
        medians = [np.median(timestamps[labels == u]) for u in unique]
        rank = {u: r for r, u in enumerate(unique[np.argsort(medians, kind="stable")])}
        return np.asarray([rank[label] for label in labels], dtype=int)

    def _compute_order(self, raw_signals: RawSignals) -> np.ndarray:
        timestamps = np.asarray(raw_signals.get_timestamps(), dtype=float)
        self.cluster_labels_ = self._cluster(timestamps)
        self.n_clusters_ = int(self.cluster_labels_.max()) + 1

        clusters: List[np.ndarray] = []
        for c in range(self.n_clusters_):
            idx = np.flatnonzero(self.cluster_labels_ == c)
            clusters.append(idx[np.argsort(timestamps[idx], kind="stable")])

        rng = check_random_state(self.random_state)
        n_mix = [int(np.floor(self.mixing * len(c))) for c in clusters]

        order: List[int] = []
        for c, cluster in enumerate(clusters):
            head = n_mix[c] if c > 0 else 0  # mixed with previous cluster
            tail = n_mix[c] if c < self.n_clusters_ - 1 else 0  # mixed with next
            order.extend(cluster[head : len(cluster) - tail].tolist())
            if c < self.n_clusters_ - 1:
                # Always emit the zone: even if this cluster contributes no
                # tail, the next cluster's head must not be dropped.
                nxt = clusters[c + 1]
                order.extend(self._mix(cluster[len(cluster) - tail :], nxt[: n_mix[c + 1]], rng))
        return np.asarray(order, dtype=int)

    @staticmethod
    def _mix(a: np.ndarray, b: np.ndarray, rng: np.random.RandomState) -> List[int]:
        """Interleave ``a`` and ``b`` with a linearly increasing share of ``b``."""
        n = len(a) + len(b)
        i_a = i_b = 0
        mixed: List[int] = []
        for pos in range(n):
            p_b = (pos + 0.5) / n
            take_b = i_a >= len(a) or (i_b < len(b) and rng.random_sample() < p_b)
            if take_b:
                mixed.append(int(b[i_b]))
                i_b += 1
            else:
                mixed.append(int(a[i_a]))
                i_a += 1
        return mixed
