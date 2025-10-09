from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorComplexity(NPSignalExtractor):

    def fit(self, X, fs=1000):
        return super().fit(X)

    def _transform(self, X):

        x_col_vars = np.var(X, axis=0, ddof=1)
        dx = np.diff(X, axis=0)
        ddx = np.diff(dx, axis=0)
        ddx_col_vars = np.var(ddx, axis=0, ddof=1)
        dx_col_vars = np.var(dx, axis=0, ddof=1)

        mobility = np.sqrt(dx_col_vars / x_col_vars)
        complexity = np.sqrt(ddx_col_vars / dx_col_vars) / mobility
        return complexity

    def attribs_per_column(self):
        return 1
