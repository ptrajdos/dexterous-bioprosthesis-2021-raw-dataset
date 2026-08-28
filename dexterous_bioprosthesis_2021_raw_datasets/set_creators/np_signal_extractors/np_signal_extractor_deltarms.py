"""Module implementing delta-RMS feature extraction.

Computes the RMS of the first-order difference of each channel.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


import numpy as np

class NpSignalExtractorDeltaRms(NPSignalExtractor):
    """Extractor that computes the RMS of the first-order difference."""

    def __init__(self, window_length_ms, offset_ms, sanitize_output=False, check_input=False, check_output=False):
        super().__init__(sanitize_output=sanitize_output, check_input=check_input, check_output=check_output)
        self.window_length_ms = window_length_ms
        self.offset_ms = offset_ms

    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        self.fs = fs
        return super().fit(X)

    def _transform(self, X):
        n_samples, n_channels = X.shape

        # Convert ms -> samples
        window_length = max(1, int(round(self.window_length_ms * self.fs / 1000)))
        offset = max(1, int(round(self.offset_ms * self.fs / 1000)))

        rms_values = []
        for start in range(0, n_samples - window_length + 1, offset):
            window = X[start:start + window_length]
            rms_values.append(np.sqrt(np.mean(window**2, axis=0)))

        rms_values = np.asarray(rms_values)

        if len(rms_values) < 2:
            return np.zeros(n_channels)

        # Mean absolute change in RMS between consecutive windows
        delta_rms = np.mean(np.abs(np.diff(rms_values, axis=0)), axis=0)

        return delta_rms

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1