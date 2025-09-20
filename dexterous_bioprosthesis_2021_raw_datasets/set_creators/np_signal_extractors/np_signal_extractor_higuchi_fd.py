
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import NPSignalExtractor

import numpy as np
import antropy as ant

class NpSignalExtractorHiguchiFD(NPSignalExtractor):

    def __init__(self, kmax =3 ) -> None:
        super().__init__()
        self.kmax = kmax
    
    def _transform(self, X):

        n_channels = X.shape[1]
        dtype = X.dtype

        attribs = []

        for channel_id in range(n_channels):
            
            x = np.copy(X[:,channel_id])
            base = ant.higuchi_fd( x , kmax=self.kmax )
            
            attribs+=[base]

        attribs = np.asanyarray(attribs, dtype=dtype)
        return attribs
    
    def attribs_per_column(self):
        return  1