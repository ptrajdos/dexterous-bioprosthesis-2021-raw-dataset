
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import NPSignalExtractor

import antropy as ant

class NpSignalExtractorPetrosianFD(NPSignalExtractor):

    def __init__(self,) -> None:
        super().__init__()
    
    
    def _transform(self, X):

        attribs = ant.petrosian_fd(X, axis=0 ).astype(X.dtype)

        return attribs
    
    def attribs_per_column(self):
        return  1