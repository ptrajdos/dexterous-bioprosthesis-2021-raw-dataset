
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import NPSignalExtractor

import numpy as np
from scipy.stats import skew


class NpSignalExtractorSkew(NPSignalExtractor):
    
    def fit(self, X):
        return self
    
    def transform(self, X):

        return skew(X)

    def attribs_per_column(self):
        return 1