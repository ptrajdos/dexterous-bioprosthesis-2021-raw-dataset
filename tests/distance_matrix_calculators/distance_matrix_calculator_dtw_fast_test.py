
from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator import DistanceMatrixCalculator
from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator_dtw_fast import DistanceMatrixCalculatorDTWFast
from tests.distance_matrix_calculators.distance_matrix_calculator_test import DistanceMatrixCalculatorTest


class DistanceMatrixCalculatorDTWFastTest(DistanceMatrixCalculatorTest):

    __test__ = True

    def get_calculator(self) -> DistanceMatrixCalculator:
        return DistanceMatrixCalculatorDTWFast(dtw_options={"dist":2, "radius":self.test_samples_number + 1})