"""Module implementing a dummy (identity) set creator.

Converts raw signals directly to numpy arrays without feature extraction.
"""
import numpy as np
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import (
    SetCreator,
)


class SetCreatorDummy(SetCreator):

    """Identity set creator that converts raw signals directly to numpy arrays."""
    def __init__(self) -> None:
        super().__init__()
        self.channel_attribs_indices = None

    def fit(self, raw_signals: RawSignals, y=None):
        """Fit the transformer to the given data."""
        self.channel_attribs_indices = []
        return super().fit(raw_signals)

    def transform(self, raw_signals: RawSignals):
        """Transform the given data."""
        if self.get_channel_attribs_indices() is None:
            raise NotFittedError("SetCreator has not been fitted.")

        X = raw_signals
        y = np.asanyarray([rs.get_label() for rs in raw_signals])
        t = np.asanyarray([rs.get_timestamp() for rs in raw_signals])
        return X, y, t

    def fit_transform(self, raw_signals: RawSignals, y=None):
        """Fit and then transform the given data."""
        self.fit(raw_signals)

        return self.transform(raw_signals)

    def get_channel_attribs_indices(self):
        """Return the feature indices grouped by channel."""
        return self.channel_attribs_indices
