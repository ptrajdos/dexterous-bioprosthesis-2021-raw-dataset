
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import NPSignalExtractor

import numpy as np
import antropy as ant

class NpSignalExtractorApEn(NPSignalExtractor):

    def __init__(self, order=2,metric='chebyshev' ) -> None:
        super().__init__()
        self.order = order
        self.metric = metric
    
    
    def _transform(self, X):

        n_channels = X.shape[1]
        dtype = X.dtype

        attribs = []

        for channel_id in range(n_channels):
            base = ant.app_entropy(X[:,channel_id], order= self.order, metric= self.metric)
            attribs+=[base]

        attribs = np.asanyarray(attribs, dtype=dtype)
        return attribs
    
    def attribs_per_column(self):
        return  1