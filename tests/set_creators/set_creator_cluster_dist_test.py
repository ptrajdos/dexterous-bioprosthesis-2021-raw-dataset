import unittest
from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator_dtw_fast2 import DistanceMatrixCalculatorDTWFast2
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_cluster_dist import SetCreatorClusterDist
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_mds import euclid_flatten
from tests.set_creators.set_creator_test import SetCreatorTest

import warnings


class SetCreatorClusterDistTest(SetCreatorTest):
    
    __test__ = True
    def get_creators(self) :
        extractors = [
            SetCreatorClusterDist(distance_calculator= DistanceMatrixCalculatorDTWFast2()),
            SetCreatorClusterDist(distance_calculator= DistanceMatrixCalculatorDTWFast2(), flatten_function=euclid_flatten)
            ]
        return extractors


if __name__ == '__main__':

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        unittest.main()