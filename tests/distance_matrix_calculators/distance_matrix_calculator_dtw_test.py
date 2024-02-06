
from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator_dtw import DistanceMatrixCalculatorDTW
from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator import DistanceMatrixCalculator
from tests.distance_matrix_calculators.distance_matrix_calculator_test import DistanceMatrixCalculatorTest


class DistanceMatrixCalculatorDTWTest(DistanceMatrixCalculatorTest):

    __test__ = True

    def get_calculator(self) -> DistanceMatrixCalculator:
        return DistanceMatrixCalculatorDTW()