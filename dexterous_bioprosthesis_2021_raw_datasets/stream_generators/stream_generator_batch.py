"""Module implementing the common batching logic of raw signal streams.

Provides :class:`AStreamGeneratorBatch`, an abstract stream that serves
signals in a fixed order (defined by subclasses) as consecutive batches,
either of a fixed size or as a fixed number of batches.
"""
from __future__ import annotations

import abc
from numbers import Integral
from typing import Optional

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.istream_generator import (
    Chunk,
    IStreamGenerator,
)


class AStreamGeneratorBatch(IStreamGenerator):
    """Stream serving signals in batches, in an order given by subclasses.

    Subclasses implement :meth:`_compute_order`. It is called once, in the
    constructor, so the stream order is fixed for the lifetime of the object
    (``reset`` replays the same stream). Subclasses must set their own
    parameters before calling ``super().__init__``.
    """

    def __init__(
        self,
        raw_signals: RawSignals,
        n_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Create the stream.

        Arguments:
        ---------
        raw_signals -- signals to be streamed
        n_batches -- number of batches; signals are split as evenly as
            possible (batch sizes differ by at most one)
        batch_size -- number of signals per batch; the last batch may be
            smaller

        At most one of ``n_batches`` and ``batch_size`` may be given.
        If neither is given, each batch contains a single signal.

        """
        if n_batches is not None and batch_size is not None:
            raise ValueError("Specify either n_batches or batch_size, not both.")

        n_signals = len(raw_signals)
        if n_signals == 0:
            raise ValueError("Cannot create a stream from empty RawSignals.")

        if n_batches is not None:
            self._check_positive_int(n_batches, "n_batches")
            if n_batches > n_signals:
                raise ValueError(
                    f"n_batches ({n_batches}) is greater than the number of signals ({n_signals})."
                )
        else:
            batch_size = 1 if batch_size is None else batch_size
            self._check_positive_int(batch_size, "batch_size")
            n_batches = int(np.ceil(n_signals / batch_size))

        self.raw_signals = raw_signals
        self.n_batches = n_batches
        self.batch_size = batch_size

        order = np.asarray(self._compute_order(raw_signals), dtype=int)
        if sorted(order.tolist()) != list(range(n_signals)):
            raise ValueError("_compute_order must return a permutation of signal indices.")
        self.order_ = order

        if batch_size is not None:
            self._batch_indices = [
                order[i : i + batch_size] for i in range(0, n_signals, batch_size)
            ]
        else:
            self._batch_indices = np.array_split(order, n_batches)

        self.n_chunks = len(self._batch_indices)
        self.classes_ = np.unique(raw_signals.get_labels())
        self.reset()

    @abc.abstractmethod
    def _compute_order(self, raw_signals: RawSignals) -> np.ndarray:
        """Return the stream order as a permutation of signal indices."""
        raise NotImplementedError

    @staticmethod
    def _check_positive_int(value, name: str) -> None:
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer. Got: {value!r}")

    def reset(self) -> None:
        """Rewind the stream to its beginning."""
        self.chunk_id = -1
        self.previous_chunk = None
        self.current_chunk = None

    def is_dry(self) -> bool:
        """Return True if all chunks have been returned."""
        return self.chunk_id + 1 >= self.n_chunks

    def get_chunk(self) -> Chunk:
        """Return the next batch as ``(RawSignals, labels)``."""
        if self.is_dry():
            raise StopIteration("The stream is dry.")

        self.chunk_id += 1
        indices = [int(i) for i in self._batch_indices[self.chunk_id]]
        X = self.raw_signals[indices]
        y = X.get_labels()

        self.previous_chunk = self.current_chunk
        self.current_chunk = (X, y)
        return self.current_chunk
