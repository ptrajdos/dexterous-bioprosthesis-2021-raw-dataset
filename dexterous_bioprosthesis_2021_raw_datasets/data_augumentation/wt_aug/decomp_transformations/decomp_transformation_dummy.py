from copy import deepcopy
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.idecomp_transformation import IDecompTransformation


class DecompTransformationDummy(IDecompTransformation):
    
    def transform(self, decompositions: list):
        return deepcopy(decompositions)