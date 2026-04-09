from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_dummy import DecompTransformationDummy
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_gaussian import DecompTransformationGaussian
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_multiplier_uniform import DecompTransformationMultiplierUniform
from tests.data_augumentation.wt_aug.decomp_transformations.idecomp_transformation_test import IDecompTransformationTest


class DecompTransformationGaussianTest(IDecompTransformationTest):
    
    __test__ = True

    def get_transformators(self) -> dict:
        return {
            "Base": DecompTransformationMultiplierUniform(),
            "Alter An":DecompTransformationMultiplierUniform(alter_approximation_coeffs=True),
            "wider": DecompTransformationMultiplierUniform(min_noise_perc=0.1, max_noise_perc=1.9),
        }