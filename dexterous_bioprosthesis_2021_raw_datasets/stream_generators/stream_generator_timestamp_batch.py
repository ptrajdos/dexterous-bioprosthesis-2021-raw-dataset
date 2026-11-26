"""Module implementing a simple timestamp-ordered batch stream generator.

Signals are sorted by timestamp and served in consecutive batches
(chunks), either of a fixed size or as a fixed number of batches.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.stream_generators.stream_generator_batch import (
    AStreamGeneratorBatch,
)


class StreamGeneratorTimestampBatch(AStreamGeneratorBatch):
    """Stream of RawSignals batches ordered by signal timestamps.

    Signals with equal timestamps keep their original relative order.
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
        super().__init__(raw_signals, n_batches=n_batches, batch_size=batch_size)

    def _compute_order(self, raw_signals: RawSignals) -> np.ndarray:
        return np.argsort(raw_signals.get_timestamps(), kind="stable")
