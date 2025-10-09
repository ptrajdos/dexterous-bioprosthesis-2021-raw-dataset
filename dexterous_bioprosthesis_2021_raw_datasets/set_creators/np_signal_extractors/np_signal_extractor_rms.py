from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorRms(NPSignalExtractor):

    def fit(self, X, fs=1000):
        return super().fit(X)

    def _transform(self, X):

        return np.sqrt(np.mean(X**2, axis=0))

    def attribs_per_column(self):
        return 1
