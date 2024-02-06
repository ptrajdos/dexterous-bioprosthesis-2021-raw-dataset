from copy import deepcopy
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.relabelers.relabeler import Relabeler

class RelabelerDummy(Relabeler):

    def fit(self, labels):
        return self
    
    def transform(self, labels):
        return deepcopy(labels)