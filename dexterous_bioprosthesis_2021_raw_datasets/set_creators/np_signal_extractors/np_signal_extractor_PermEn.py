
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import NPSignalExtractor

import numpy as np
import antropy as ant

class NpSignalExtractorPermEn(NPSignalExtractor):

    def __init__(self, order=3, delay=1, normalize=False ) -> None:
        super().__init__()
        self.order = order
        self.delay = delay
        self.normalize = normalize

    
    def _transform(self, X):

        n_channels = X.shape[1]
        dtype = X.dtype

        attribs = []

        for channel_id in range(n_channels):
            base = ant.perm_entropy(X[:,channel_id], order= self.order, delay=self.delay, 
                                    normalize=self.normalize)
            attribs+=[base]

        attribs = np.asanyarray(attribs, dtype=dtype)
        return attribs
    
    def attribs_per_column(self):
        return  1