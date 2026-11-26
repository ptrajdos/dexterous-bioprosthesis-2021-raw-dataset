"""Module implementing a feature-extracting stream wrapper.

Wraps any :class:`IStreamGenerator` producing ``RawSignals`` chunks and
turns each chunk into a numeric ``(X, y)`` pair using a
:class:`~dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator.SetCreator`,
so the stream can be consumed by stream-learn evaluators and classifiers.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import SetCreator
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.istream_generator import (
    IStreamGenerator,
)


class StreamGeneratorFeatureExtractor(IStreamGenerator):
    """Stream wrapper extracting features from raw-signal chunks.

    The set creator is fitted on the first chunk only; every later chunk is
    transformed with that fitted model. The set creator passed to the
    constructor is not modified: a copy is fitted and stored as
    ``set_creator_``. Calling :meth:`reset` discards the fitted copy, so the
    next pass over the stream fits again on its first chunk.

    Note: with small first chunks (e.g. the default one-signal batches),
    data-dependent set creators are fitted on very little data.
    """

    def __init__(self, stream: IStreamGenerator, set_creator: SetCreator) -> None:
        """Create the wrapper.

        Arguments:
        ---------
        stream -- stream returning ``(RawSignals, labels)`` chunks
        set_creator -- set creator used to extract features

        """
        self.stream = stream
        self.set_creator = set_creator
        self.reset()

    # Chunk counters and classes come from the wrapped stream, so both
    # always agree on the stream position.
    @property
    def n_chunks(self) -> int:
        return self.stream.n_chunks

    @property
    def chunk_id(self) -> int:
        return self.stream.chunk_id

    @property
    def classes_(self) -> np.ndarray:
        return self.stream.classes_

    def reset(self) -> None:
        """Rewind the wrapped stream and forget the fitted set creator."""
        self.stream.reset()
        self.set_creator_: Optional[SetCreator] = None
        self.previous_chunk = None
        self.current_chunk = None

    def is_dry(self) -> bool:
        """Return True if the wrapped stream is dry."""
        return self.stream.is_dry()

    def get_chunk(self):
        """Return the next chunk as ``(X, y)`` with extracted features."""
        raw_X, _ = self.stream.get_chunk()

        if self.set_creator_ is None:
            self.set_creator_ = deepcopy(self.set_creator)
            X, y, _ = self.set_creator_.fit_transform(raw_X)
        else:
            X, y, _ = self.set_creator_.transform(raw_X)

        self.previous_chunk = self.current_chunk
        self.current_chunk = (X, y)
        return self.current_chunk
