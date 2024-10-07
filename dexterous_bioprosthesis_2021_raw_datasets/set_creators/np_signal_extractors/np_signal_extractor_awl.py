
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import NPSignalExtractor

import numpy as np


class NpSignalExtractorAWL(NPSignalExtractor):
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        differences = np.abs(np.diff(X, axis=0))
    
        awl = np.mean(differences, axis=0)

        return awl

    def attribs_per_column(self):
        return 1