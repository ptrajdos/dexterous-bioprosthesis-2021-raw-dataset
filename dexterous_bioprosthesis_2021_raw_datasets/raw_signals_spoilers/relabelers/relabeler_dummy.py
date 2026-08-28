"""Module implementing identity (dummy) relabeling.

Returns signals with their original labels unchanged.
"""
from copy import deepcopy
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.relabelers.relabeler import Relabeler

class RelabelerDummy(Relabeler):
    """Identity relabeler that preserves original signal labels."""

    def fit(self, labels):
        """Fit the transformer to the given data."""
        return self
    
    def transform(self, labels):
        """Transform the given data."""
        return deepcopy(labels)