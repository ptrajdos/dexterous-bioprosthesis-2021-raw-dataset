
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import NPSignalExtractor

import numpy as np
import antropy as ant

class NpSignalExtractorHiguchiFD2(NPSignalExtractor):

    def __init__(self) -> None:
        super().__init__()
    
    def _transform(self, X):

        n_channels = X.shape[1]
        dtype = X.dtype

        attribs = []

        for channel_id in range(n_channels):
            
            x = np.copy(X[:,channel_id])
            k = min(len(x)//2 - 1,1)
            base = ant.higuchi_fd( x , kmax=k )
            
            attribs+=[base]

        attribs = np.asanyarray(attribs, dtype=dtype)
        return attribs
    
    def attribs_per_column(self):
        return  1