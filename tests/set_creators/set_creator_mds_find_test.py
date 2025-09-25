import unittest
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_mds import SetCreatorMDS, euclid_flatten
from tests.set_creators.set_creator_test import SetCreatorTest

import warnings


class SetCreatorMDSFindTest(SetCreatorTest):
    
    __test__ = True
    def get_creators(self) :
        extractors = [
            SetCreatorMDS(n_attr=10,step=1, find_best=True), 
            SetCreatorMDS(n_attr=10,step=1, find_best=True, flatten_function=euclid_flatten), 
            ]
        return extractors

    

if __name__ == '__main__':

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        unittest.main()