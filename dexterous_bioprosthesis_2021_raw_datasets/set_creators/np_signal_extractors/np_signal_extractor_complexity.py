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

        dx_col_vars = np.var(dx, axis=0, ddof=1)
        ddx_col_vars = np.var(ddx, axis=0, ddof=1)

        mobility = np.zeros_like(x_col_vars)
        np.divide(
            np.sqrt(dx_col_vars),
            np.sqrt(x_col_vars),
            out=mobility,
            where=x_col_vars > 0
        )

        complexity = np.zeros_like(x_col_vars)
        np.divide(
            np.sqrt(ddx_col_vars / dx_col_vars),
            mobility,
            out=complexity,
            where=(dx_col_vars > 0) & (mobility > 0)
        )

        return complexity


    def attribs_per_column(self):
        return 1
