from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


class NpSignalExtractorAr(NPSignalExtractor):

    def __init__(
        self,
        sanitize_output=False,
        check_input=False,
        check_output=False,
        lags=2,
        ar_args={},
    ) -> None:
        super().__init__(
            sanitize_output=sanitize_output,
            check_input=check_input,
            check_output=check_output,
        )
        self.lags = lags
        self.ar_args = ar_args

    def fit(self, X):
        return super().fit(X)

    def _transform(self, X):

        n_channels = X.shape[1]

        attribs = []

        for channel_id in range(n_channels):
            ch_series = pd.Series(X[:, channel_id])
            model = AutoReg(ch_series, lags=self.lags, **self.ar_args).fit()
            m_params = [val for val in model.params]
            attribs += m_params

        attribs = np.asanyarray(attribs).astype(X.dtype)
        return attribs

    def attribs_per_column(self):
        return self.lags + 1
