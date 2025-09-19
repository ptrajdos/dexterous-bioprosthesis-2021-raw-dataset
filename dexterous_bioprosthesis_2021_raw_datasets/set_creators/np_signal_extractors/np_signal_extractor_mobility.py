from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorMobility(NPSignalExtractor):

    def fit(self, X):
        return super().fit(X)

    def _transform(self, X):

        x_col_vars = np.var(X, axis=0, ddof=1)
        dx = np.diff(X, axis=0)
        dx_col_vars = np.var(dx, axis=0, ddof=1)

        mobility = np.sqrt(dx_col_vars / x_col_vars)
        return mobility

    def attribs_per_column(self):
        return 1
