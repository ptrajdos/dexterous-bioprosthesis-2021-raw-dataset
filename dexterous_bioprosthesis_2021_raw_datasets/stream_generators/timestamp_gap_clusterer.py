"""Module implementing 1D gap-based clustering of timestamps.

The number of clusters is detected automatically: timestamps are sorted
and a new cluster starts wherever the gap to the previous timestamp is an
outlier among all gaps (Tukey's rule, ``gap > Q3 + iqr_factor * IQR``).
This suits recordings made in separate sessions, where gaps between
sessions are much larger than gaps between signals within a session.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class TimestampGapClusterer(ClusterMixin, BaseEstimator):
    """Cluster 1D values (timestamps) by splitting at unusually large gaps.

    Parameters
    ----------
    iqr_factor : float
        Multiplier of the gap IQR in the outlier rule. Larger values give
        fewer clusters.
    min_gap : float
        Gaps not larger than this value never split clusters. Useful when
        most gaps are zero (many equal timestamps), which makes the IQR
        rule split at any positive gap.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Cluster label of each input value. Labels are ordered in time:
        cluster 0 contains the earliest timestamps.
    n_clusters_ : int
        Number of detected clusters.
    threshold_ : float
        Gap threshold actually used for splitting.

    """

    def __init__(self, iqr_factor: float = 3.0, min_gap: float = 0.0) -> None:
        self.iqr_factor = iqr_factor
        self.min_gap = min_gap

    def fit(self, X, y=None):
        """Detect clusters.

        Arguments:
        ---------
        X -- array of shape (n_samples,) or (n_samples, 1) with timestamps
        y -- ignored

        """
        values = np.asarray(X, dtype=float)
        if values.ndim == 2:
            if values.shape[1] != 1:
                raise ValueError(f"Expected one feature, got {values.shape[1]}.")
            values = values[:, 0]
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Expected a non-empty 1D array of timestamps.")
        if self.iqr_factor < 0:
            raise ValueError(f"iqr_factor must be non-negative. Got {self.iqr_factor}")
        if self.min_gap < 0:
            raise ValueError(f"min_gap must be non-negative. Got {self.min_gap}")

        order = np.argsort(values, kind="stable")
        gaps = np.diff(values[order])

        if gaps.size == 0:
            threshold = np.inf
        else:
            q1, q3 = np.percentile(gaps, [25, 75])
            threshold = max(q3 + self.iqr_factor * (q3 - q1), self.min_gap)

        sorted_labels = np.concatenate([[0], np.cumsum(gaps > threshold)])
        labels = np.empty_like(sorted_labels)
        labels[order] = sorted_labels

        self.labels_ = labels.astype(int)
        self.n_clusters_ = int(sorted_labels[-1]) + 1
        self.threshold_ = float(threshold)
        return self
