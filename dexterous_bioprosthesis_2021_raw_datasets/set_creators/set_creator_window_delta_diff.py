"""Module implementing window-difference-based feature extraction.

Extracts features by computing differences of attributes between consecutive windows.
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


class SetCreatorWindowDeltaDiff(ASetCreator):
    """Set creator that constructs features by differentiating attributes between consecutive windows.

    All signals must have the same length, so that every signal yields the
    same number of windows.
    """

    def __init__(self, extractors=[], window_length=0.33, overlap=0.5) -> None:
        """A SetCreator that applies a list of functions to windowed raw signals and computes differences between consecutive windows.

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
        self.signal_length = len(raw_signals[0])
        self._check_lengths(raw_signals)
        n_extractors = len(self.extractors)

        self.segmenter = RawSignalsFilterWindowSegmentation(
            window_length=self.window_length, overlap=self.overlap
        )
        self.segmenter.fit(raw_signals=raw_signals)
        subset = raw_signals[0:1]
        tmp_segmented = self.segmenter.transform(raw_signals=subset)
        self.n_windows = len(tmp_segmented)
        self.n_deltas = max(0, self.n_windows - 1)

        self.channel_selected_attribs = []
        for ch_id in range(self.n_channels):
            self.channel_selected_attribs.append([])

        offset = 0
        for extr_id in range(n_extractors):
            n_attribs_per_channel = self.extractors[extr_id].attribs_per_column()
            for delta_idx in range(self.n_deltas):
                for ch_id in range(self.n_channels):
                    tmp_attrib_idxs = [offset + i for i in range(n_attribs_per_channel)]
                    self.channel_selected_attribs[ch_id] += tmp_attrib_idxs
                    offset += n_attribs_per_channel

        self._num_attribs = offset

        return self

    def _check_lengths(self, raw_signals: RawSignals):
        """Raise ValueError unless all signals have the fitted length."""
        for raw_signal in raw_signals:
            if len(raw_signal) != self.signal_length:
                raise ValueError(
                    "All signals must have the same length ({} samples, as in fit). Got a signal with {} samples. "
                    "Standardise signal lengths first, e.g. with RawSignalsFilterLengthMedianStandarizer.".format(
                        self.signal_length, len(raw_signal)
                    )
                )

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        if self.get_channel_attribs_indices() is None:
            raise NotFittedError("SetCreator has not been fitted.")

        raw_signals = RawSignals.construct_from_list(raw_signals)
        self._check_lengths(raw_signals)

        n_signals = len(raw_signals)
        extracted_attribs = np.zeros((n_signals, self._num_attribs))
        labels = []
        timestamps = []

        for raw_signal_id, raw_signal in enumerate(raw_signals):

            labels.append(raw_signal.get_label())
            timestamps.append(raw_signal.get_timestamp())
            tmp_signals = raw_signals.initialize_empty()
            tmp_signals.append(raw_signal)
            segmented_signals = self.segmenter.transform(tmp_signals)

            offset = 0
            for extractor_id, extractor in enumerate(self.extractors):
                # Extract attributes for each window
                window_attribs = []
                for segmented_signal in segmented_signals:
                    signal = segmented_signal.to_numpy()
                    extracted = extractor.fit_transform(
                        signal, fs=segmented_signal.get_sample_rate()
                    )
                    window_attribs.append(extracted)

                # Compute differences between consecutive windows
                for i in range(len(window_attribs) - 1):
                    delta = window_attribs[i + 1] - window_attribs[i]
                    n_delta = delta.shape[0]
                    extracted_attribs[raw_signal_id, offset : (offset + n_delta)] = delta
                    offset += n_delta

        extracted_attribs = extracted_attribs.astype(self._output_dtype(raw_signals))
        labels = np.asanyarray(labels)
        timestamps = np.asanyarray(timestamps)
        return self._wrap_output(extracted_attribs), labels, timestamps

    def fit_transform(self, raw_signals: RawSignals, y=None):
        """Fit and then transform the given data."""
        return self.fit(raw_signals).transform(raw_signals)

    def get_channel_attribs_indices(self):
        """Return the feature indices grouped by channel."""
        return self.channel_selected_attribs

    def get_feature_names_out(self, input_features=None):
        """Return attribute names formatted as {channel}_WD{i}_{attribute}.
        WD{i} is the difference between windows i+1 and i.
        """
        if self.get_channel_attribs_indices() is None:
            raise NotFittedError("SetCreator has not been fitted.")

        return self._feature_names(
            ["WD{}".format(i) for i in range(self.n_deltas)],
            [extractor.get_attribute_names() for extractor in self.extractors],
        )
