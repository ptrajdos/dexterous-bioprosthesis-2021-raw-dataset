import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import NPSignalExtractor


class NpSignalExtractorOffsetRatio(NPSignalExtractor):
    """
    Offset relative to amplitude: |mean| / mean(|x|).
    Amplitude invariant. Close to 0 for zero-mean EMG, grows with baseline drift.
    """

    def __init__(self, eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def fit(self, X, fs=1000):
        """Fit the transformer to the given data."""
        return super().fit(X, fs=fs)

    def _transform(self, X):
        return np.abs(np.mean(X, axis=0)) / (np.mean(np.abs(X), axis=0) + self.eps)

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
