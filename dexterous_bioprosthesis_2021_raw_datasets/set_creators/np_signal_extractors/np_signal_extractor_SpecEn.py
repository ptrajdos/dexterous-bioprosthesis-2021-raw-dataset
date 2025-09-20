
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import NPSignalExtractor

import numpy as np
import antropy as ant

class NpSignalExtractorSpecEn(NPSignalExtractor):

    def __init__(self, sf=1000,method='fft', nperseg=None, normalize=False, axis=0 ) -> None:
        super().__init__()
        self.sf = sf
        self.method = method
        self.nperseg = nperseg
        self.normalize = normalize
        self.axis = axis
    
    
    def _transform(self, X):

        attribs = ant.spectral_entropy(X, sf= self.sf, method= self.method, nperseg=self.nperseg,
                                    normalize=self.normalize, axis=self.axis)
        
        attribs = np.asanyarray(attribs, dtype=X.dtype)
        return attribs
    
    def attribs_per_column(self):
        return  1