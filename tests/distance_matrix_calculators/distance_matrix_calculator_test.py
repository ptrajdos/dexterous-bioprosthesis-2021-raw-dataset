import unittest

from dexterous_bioprosthesis_2021_raw_datasets.distance_matrix_calculators.distance_matrix_calculator import DistanceMatrixCalculator
from tests.testing_tools import generate_sample_data
import numpy as np
import random
import itertools
import platform

class DistanceMatrixCalculatorTest(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.test_samples_number = 12

    __test__ = False
    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise  unittest.SkipTest("Skipping")

    def get_calculator(self)->DistanceMatrixCalculator:
        raise  unittest.SkipTest("Skipping")

    def test_distance_calculator(self):
        calculator = self.get_calculator()

        signals = generate_sample_data(samples_number=self.test_samples_number)
        n_signals = len(signals)
        n_channels = len(signals[0].channel_names)

        distance_matrix = calculator.calculate_distance_matrix(signals)

        self.assertIsNotNone(distance_matrix, "Distance matrix cannot be none!")
        self.assertFalse( np.any( np.isnan( distance_matrix )), "No nan's are allowed in the distance matrix!")
        self.assertFalse( np.any( np.isinf( distance_matrix))  , "No infinity entries are allowed in the distance matrix" )
        self.assertTrue( distance_matrix.shape == ( n_channels , n_signals, n_signals), "Dimmensionality of the matrix is incorrect")
        self.assertFalse(np.any(distance_matrix < 0), "No negative distances are allowed")

        for ch in range(n_channels):

            self.assertTrue( 
                np.allclose( distance_matrix[ch,:,:], distance_matrix[ch,:,:].T,rtol=1E-5,atol=1E-8 ),
                 "Distance matrix is not symmetric" 
                 )
            
            self.assertFalse(  np.any(  np.diag( distance_matrix[ch,: , :] ) >0 ) , "Nonzero value" ) 
            self.assertLess( np.sum( np.diag( distance_matrix[ch,: , :] )), 1E-8, "Diagonal must contain only zeros" )

            #Triangle inequality
            # self.assertTrue( self.check_triangle_inequality(distance_matrix[ch,:,:]) )

    def check_triangle_inequality(self,matrix):
        """ Returns true iff the matrix D fulfills
        the triangle inequaltiy.
        """
        n = len(matrix)
        valid = True
        for i in range(n):
            for j in range(i, n):
                for k in range(n):
                    if k == j or k == i:
                        continue
                    if matrix[i][j] >= matrix[i][k] + matrix[k][j]:
                        return False
        return valid
    
    def test_distance_calculator_2_sets(self):
        calculator = self.get_calculator()

        signals = generate_sample_data(samples_number=self.test_samples_number)
        n_signals = len(signals)
        n_channels = len(signals[0].channel_names)

        distance_matrix = calculator.calculate_distance_matrix_set_2_set(signals, signals)

        self.assertIsNotNone(distance_matrix, "Distance matrix cannot be none!")
        self.assertFalse( np.any( np.isnan( distance_matrix )), "No nan's are allowed in the distance matrix!")
        self.assertFalse( np.any( np.isinf( distance_matrix))  , "No infinity entries are allowed in the distance matrix" )
        self.assertTrue( distance_matrix.shape == ( n_channels , n_signals, n_signals), "Dimmensionality of the matrix is incorrect")
        self.assertFalse(np.any(distance_matrix < 0), "No negative distances are allowed")

        for ch in range(n_channels):

            self.assertTrue( 
                np.allclose( distance_matrix[ch,:,:], distance_matrix[ch,:,:].T,rtol=1E-5,atol=1E-8 ),
                 "Distance matrix is not symmetric" 
                 )
            
           
            self.assertFalse(  np.any(  np.diag( distance_matrix[ch,: , :] ) >0 ) , "Nonzero value" ) 
            self.assertLess( np.sum( np.diag( distance_matrix[ch,: , :] )), 1E-8, "Diagonal must contain only zeros" )

    def test_distance_calculator_2_set(self):
        calculator = self.get_calculator()

        signals = generate_sample_data(samples_number=self.test_samples_number)
        n_signals = len(signals)
        n_channels = len(signals[0].channel_names)

        ref_signal = signals[0]

        distance_matrix = calculator.raw_signal_dist_2_set(ref_signal,signals)

        self.assertIsNotNone(distance_matrix, "Distance matrix cannot be none!")
        self.assertFalse( np.any( np.isnan( distance_matrix )), "No nan's are allowed in the distance matrix!")
        self.assertFalse( np.any( np.isinf( distance_matrix))  , "No infinity entries are allowed in the distance matrix" )
        self.assertTrue( distance_matrix.shape == ( n_channels , n_signals), "Dimmensionality of the matrix is incorrect")
        self.assertFalse(np.any(distance_matrix < 0), "No negative distances are allowed")


if __name__ == '__main__':
    unittest.main()