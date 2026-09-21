"""Module implementing a conditional column filter.

Applies a given filter only to signal columns whose names match a regular
expression, leaving the remaining columns unchanged.
"""
from __future__ import annotations

import re
from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterConditionalColumn(RawSignalsFilter):
    """Filter that conditionally applies filters to columns matching regexes.

    Arguments:
    ---------
    column_filter_pairs -- a single tuple/list of (column_regex, filter) or
        a list of such pairs.  Each pair consists of a regular expression
        string used to select columns and a :class:`RawSignalsFilter`
        instance to apply to the matched columns.  When multiple pairs are
        given they are applied sequentially.

    """

    def __init__(self, column_filter_pairs) -> None:
        super().__init__()
        # Accept a single pair (regex, filter) or a list of pairs.
        if (isinstance(column_filter_pairs, (tuple, list))
                and len(column_filter_pairs) == 2
                and isinstance(column_filter_pairs[0], str)):
            column_filter_pairs = [column_filter_pairs]
        self.column_filter_pairs = list(column_filter_pairs)

    def _split_signals(self, raw_signals: RawSignals):
        compiled = self._current_compiled
        matched_signals = RawSignals()
        indices_list = []

        for raw_signal in raw_signals:
            channel_names = raw_signal.channel_names
            matching_indices = [i for i, name in enumerate(channel_names) if compiled.search(name)]
            non_matching_indices = [i for i in range(len(channel_names)) if i not in matching_indices]
            indices_list.append((matching_indices, non_matching_indices))

            matched_signal = RawSignal(
                signal=raw_signal.signal[:, matching_indices] if matching_indices else np.empty((raw_signal.signal.shape[0], 0), dtype=raw_signal.signal.dtype),
                object_class=deepcopy(raw_signal.object_class),
                channel_names=[channel_names[i] for i in matching_indices],
                timestamp=raw_signal.timestamp,
                sample_rate=raw_signal.sample_rate,
            )
            matched_signals.append(matched_signal)

        return matched_signals, indices_list

    def _merge_signals(self, raw_signals: RawSignals, transformed_matched: RawSignals, indices_list):
        result = RawSignals()

        for raw_signal, transformed, (matching_indices, non_matching_indices) in zip(raw_signals, transformed_matched, indices_list):
            n_rows = raw_signal.signal.shape[0]
            n_cols = len(matching_indices) + len(non_matching_indices)
            merged = np.empty((n_rows, n_cols), dtype=raw_signal.signal.dtype)
            merged_names = [''] * n_cols

            for new_idx, orig_idx in enumerate(matching_indices):
                merged[:, orig_idx] = transformed.signal[:, new_idx]
                merged_names[orig_idx] = transformed.channel_names[new_idx]

            for orig_idx in non_matching_indices:
                merged[:, orig_idx] = raw_signal.signal[:, orig_idx]
                merged_names[orig_idx] = raw_signal.channel_names[orig_idx]

            result.append(RawSignal(
                signal=merged,
                object_class=deepcopy(raw_signal.object_class),
                channel_names=merged_names,
                timestamp=raw_signal.timestamp,
                sample_rate=raw_signal.sample_rate,
            ))

        return result

    def _apply_single(self, raw_signals: RawSignals, column_regex, inner_filter, fit=False, y=None):
        compiled = re.compile(column_regex)
        self._current_compiled = compiled
        matched_signals, indices_list = self._split_signals(raw_signals)
        if fit:
            inner_filter.fit(matched_signals, y)
        transformed_matched = inner_filter.transform(matched_signals)
        return self._merge_signals(raw_signals, transformed_matched, indices_list)

    def _has_matches(self, indices_list):
        return any(len(matching) > 0 for matching, _ in indices_list)

    def fit(self, raw_signals: RawSignals, y=None):
        """Fits all inner filters on their matched columns."""
        current = raw_signals
        for regex, inner_filter in self.column_filter_pairs:
            compiled = re.compile(regex)
            self._current_compiled = compiled
            matched_signals, indices_list = self._split_signals(current)
            if not self._has_matches(indices_list):
                continue
            inner_filter.fit(matched_signals, y)
            # Apply transform so subsequent filters see the updated signals
            transformed_matched = inner_filter.transform(matched_signals)
            current = self._merge_signals(current, transformed_matched, indices_list)
        return super().fit(raw_signals, y)

    def transform(self, raw_signals: RawSignals):
        """Transforms columns sequentially for each regex/filter pair."""
        self._check_fitted()
        current = raw_signals
        for regex, inner_filter in self.column_filter_pairs:
            compiled = re.compile(regex)
            self._current_compiled = compiled
            matched_signals, indices_list = self._split_signals(current)
            if not self._has_matches(indices_list):
                continue
            transformed_matched = inner_filter.transform(matched_signals)
            current = self._merge_signals(current, transformed_matched, indices_list)
        return current
