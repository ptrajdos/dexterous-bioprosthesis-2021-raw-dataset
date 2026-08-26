"""Module implementing constant relabeling.

Assigns a fixed label value to all signals.
"""
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.relabelers.relabeler import Relabeler

class RelabelerConstant(Relabeler):
    
    """Relabeler that assigns a fixed label to all signals."""
    def __init__(self, new_label=0) -> None:
        super().__init__()

        self.new_label = new_label

    def fit(self, labels):
        """Fit the transformer to the given data."""
        return self
    
    def transform(self, labels):
        """Transform the given data."""
        n_labels = len(labels)

        new_labels = [self.new_label for _ in range(n_labels)]

        return new_labels