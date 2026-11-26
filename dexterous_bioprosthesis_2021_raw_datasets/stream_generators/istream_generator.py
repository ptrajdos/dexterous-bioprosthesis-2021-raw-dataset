"""Module defining the abstract interface for raw signal stream generators.

Provides :class:`IStreamGenerator`, a contract compatible with
``strlearn.streams.DataStream`` (https://github.com/w4k2/stream-learn).
The abstract part (``get_chunk``, ``is_dry``, ``__next__``, ``__iter__``)
mirrors ``DataStream``. On top of it, the attributes that stream-learn
evaluators (e.g. ``TestThenTrain``) read from a stream are declared:
``n_chunks``, ``chunk_id``, ``previous_chunk`` and ``classes_``.

A chunk is a tuple ``(X, y)`` where ``X`` is a
:class:`~dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals.RawSignals`
object and ``y`` is a numpy array with the labels of the signals in ``X``.
"""
from __future__ import annotations

import abc
from typing import Iterator, Optional, Tuple

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals

Chunk = Tuple[RawSignals, np.ndarray]


class IStreamGenerator(abc.ABC):
    """Interface for streams of raw signals.

    Attributes
    ----------
    n_chunks : int
        Total number of chunks in the stream.
    chunk_id : int
        Index of the most recently returned chunk (-1 before the first chunk).
    previous_chunk : tuple or None
        The chunk returned before the current one, None for the first chunk.
    current_chunk : tuple or None
        The most recently returned chunk.
    classes_ : numpy.ndarray
        All classes present in the whole stream.

    """

    n_chunks: int
    chunk_id: int
    previous_chunk: Optional[Chunk]
    current_chunk: Optional[Chunk]
    classes_: np.ndarray

    @abc.abstractmethod
    def get_chunk(self) -> Chunk:
        """Return the next chunk of the stream as ``(X, y)``."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_dry(self) -> bool:
        """Return True if there are no more chunks to return."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Rewind the stream to its beginning."""
        raise NotImplementedError

    def __next__(self) -> Iterator[Chunk]:
        # Same (generator-returning) semantics as strlearn's DataStream.
        while not self.is_dry():
            yield self.get_chunk()

    def __iter__(self) -> Iterator[Chunk]:
        return next(self)
