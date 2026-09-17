"""Module implementing concatenated set creation from multiple extractors.

Combines features from multiple set creators by concatenation.
"""
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import ASetCreator

import pandas as pd
import numpy as np


class SetCreatorConcatenated(ASetCreator):
    """Set creator that concatenates features from multiple sub-creators."""

    def __init__(self, creators) -> None:
        super().__init__()

        self.creators = creators
        self.channel_selected_attribs = None # List containing number of attributes for each channel
        self._concatenated_channel_names = tuple()

    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        super().fit(raw_signals, y)
        raw_signals = RawSignals.construct_from_list(raw_signals)
        for creator in self.creators:
            creator.fit(raw_signals)

        # Concatenate channel names from all sub-creators
        self._concatenated_channel_names = []
        for creator in self.creators:
            creator_names = creator.get_channel_names()
            for name in creator_names:
                if name not in self._concatenated_channel_names:
                    self._concatenated_channel_names.append(name)
        self._concatenated_channel_names = tuple(self._concatenated_channel_names)

        # Build concatenated attribute indices
        is_creators_channel_attribs = [creator.get_channel_attribs_indices() is not None for creator in self.creators]
        if all(is_creators_channel_attribs):
            # Build a mapping from channel name to index in the concatenated channel list
            channel_name_to_idx = {name: idx for idx, name in enumerate(self._concatenated_channel_names)}
            n_concat_channels = len(self._concatenated_channel_names)
            self.channel_selected_attribs = [[] for _ in range(n_concat_channels)]

            offset = 0
            for creator in self.creators:
                creator_channel_attribs = creator.get_channel_attribs_indices()
                creator_names = creator.get_channel_names()
                for creator_ch_idx, attrib_indices in enumerate(creator_channel_attribs):
                    concat_ch_idx = channel_name_to_idx[creator_names[creator_ch_idx]]
                    self.channel_selected_attribs[concat_ch_idx] += list(offset + np.asanyarray(attrib_indices))
                # Compute offset as total number of features from this creator
                offset += self._count_features(creator_channel_attribs)

        return self

    def _count_features(self, channel_attribs):
        """Count total number of features across all channels."""
        count = 0
        for attrib_indices in channel_attribs:
            count += len(attrib_indices)
        return count
    

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        if self.get_channel_attribs_indices() is None:
            raise NotFittedError("SetCreator has not been fitted.")
        
        raw_signals = RawSignals.construct_from_list(raw_signals)
        X_es = []
        y_f = None
        t_f = None

        for creator in self.creators:
            X, y, t = creator.transform(raw_signals)

            if y_f is None:
                y_f = y
                t_f = t

            if isinstance(X, pd.DataFrame ):
                X = pd.DataFrame.to_numpy(X)

            X_es.append(X)
        
        Xf = np.concatenate(X_es, axis=1)
        
        return Xf, y_f, t_f
    

    def fit_transform(self, raw_signals: RawSignals, y=None):
        """Fit and then transform the given data."""
        self.fit(raw_signals)
        return self.transform(raw_signals)
    
    def get_channel_attribs_indices(self):
        """Return the feature indices grouped by channel."""
        return self.channel_selected_attribs

    def get_channel_names(self):
        """Return concatenated channel names from all sub-creators."""
        return self._concatenated_channel_names