from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np
from scipy.stats import kurtosis


class NpSignalExtractorKurtosis(NPSignalExtractor):

    def fit(self, X, fs=1000):
        return super().fit(X)

    def _transform(self, X):

        kurtosis_values = kurtosis(X, nan_policy="propagate")
        #FIXME Is there a better way to handle NaN/Inf values?
        kurtosis_values = np.nan_to_num(kurtosis_values, nan=0.0, posinf=0.0, neginf=0.0)

        return kurtosis_values
    

    def attribs_per_column(self):
        return 1
