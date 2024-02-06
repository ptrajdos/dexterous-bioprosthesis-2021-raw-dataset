import unittest
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_mds import SetCreatorMDS, euclid_flatten
from tests.set_creators.set_creator_test import SetCreatorTest

import warnings


class SetCreatorMDSTest(SetCreatorTest):
    
    __test__ = True
    def get_creators(self) :
        extractors = [SetCreatorMDS(), SetCreatorMDS(flatten_function=euclid_flatten)]
        return extractors


if __name__ == '__main__':

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        unittest.main()