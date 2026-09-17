"""Module implementing function-based feature extraction.

Extracts features using configurable numpy signal extractor functions.
"""

from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_window_segmentation import (
    RawSignalsFilterWindowSegmentation,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import (
    ASetCreator,
)
import numpy as np
import pywt


class SetCreatorWindowDelta(ASetCreator):
    """Set creator using configurable numpy signal extractor functions."""

    def __init__(self, extractors=[], window_length=0.33, overlap=0.5) -> None:
        """A SetCreator that applies a list of functions to the raw signals to create the dataset.

        Arguments:
        ---------
        extractors -- A list of functions that take a signal as input and return a vector of attributes
        window_length -- window length as fraction of signal length
        overlap  -- overlap between windows as fraction of window length

        """
        super().__init__()
        self.extractors = extractors
        self.window_length = window_length
        self.overlap = overlap

        self._num_attribs = None
        self.n_channels = None
        self.channel_selected_attribs = (
            None  # List containing number of attributes for each channel
        )

    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        super().fit(raw_signals, y)
        raw_signals = RawSignals.construct_from_list(raw_signals)

        self.n_channels = raw_signals[0].to_numpy().shape[1]
        n_objects = len(raw_signals)
        n_extractors = len(self.extractors)

        self.segmenter = RawSignalsFilterWindowSegmentation(
            window_length=self.window_length, overlap=self.overlap
        )
        self.segmenter.fit(raw_signals=raw_signals)
        subset = raw_signals[0:1]
        tmp_segmented = self.segmenter.transform(raw_signals=subset)
        self.n_windows = len(tmp_segmented)
        
        self.channel_selected_attribs = []
        for ch_id in range(self.n_channels):
            self.channel_selected_attribs.append([])

        offset = 0
        for extr_id in range(n_extractors):
            n_attribs_per_channel = self.extractors[extr_id].attribs_per_column()
            for win_idx in range(self.n_windows):
                for ch_id in range(self.n_channels):
                    tmp_attrib_idxs = [offset + i for i in range(n_attribs_per_channel)]
                    self.channel_selected_attribs[ch_id] += tmp_attrib_idxs
                    offset += n_attribs_per_channel

        self._num_attribs = offset

        return self

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        if self.get_channel_attribs_indices() is None:
            raise NotFittedError("SetCreator has not been fitted.")

        raw_signals = RawSignals.construct_from_list(raw_signals)
        
        n_signals = len(raw_signals)
        extracted_attribs = np.zeros((n_signals, self._num_attribs))
        labels = []
        timestamps = []

        for raw_signal_id, raw_signal in enumerate(raw_signals):

            
            labels.append(raw_signal.get_label())
            timestamps.append(raw_signal.get_timestamp())
            offset = 0
            for extractor_id, extractor in enumerate(self.extractors):
                tmp_signals = raw_signals.initialize_empty()
                tmp_signals.append(raw_signal)
                segmented_signals = self.segmenter.transform(tmp_signals)

                for segmented_signal in segmented_signals:
                    signal = segmented_signal.to_numpy()
                    
                    extracted = extractor.fit_transform(signal)
                    n_extracted = extracted.shape[0]
                    extracted_attribs[raw_signal_id, offset : (offset + n_extracted)] = (
                        extracted
                    )
                    offset += n_extracted

        extracted_attribs = extracted_attribs.astype(raw_signals[0].to_numpy().dtype)
        labels = np.asanyarray(labels)
        timestamps = np.asanyarray(timestamps)
        return extracted_attribs, labels, timestamps

    def fit_transform(self, raw_signals: RawSignals, y=None):
        """Fit and then transform the given data."""
        return self.fit(raw_signals).transform(raw_signals)

    def get_channel_attribs_indices(self):
        """Return the feature indices grouped by channel."""
        return self.channel_selected_attribs
