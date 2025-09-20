
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import NPSignalExtractor

import numpy as np
import antropy as ant

class NpSignalExtractorSVDEn(NPSignalExtractor):

    def __init__(self, order=2,delay=1, normalise=True ) -> None:
        super().__init__()
        self.order = order
        self.delay = delay
        self.normalise = normalise
    
    
    def _transform(self, X):

        n_channels = X.shape[1]

        attribs = []

        for channel_id in range(n_channels):
            base = ant.svd_entropy(X[:,channel_id], order= self.order, delay=self.delay, normalize= self.normalise)
            attribs+=[base]

        attribs = np.asanyarray(attribs, dtype=X.dtype)
        return attribs
    
    def attribs_per_column(self):
        return  1