from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_dummy import DecompTransformationDummy
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_gaussian import DecompTransformationGaussian
from tests.data_augumentation.wt_aug.decomp_transformations.idecomp_transformation_test import IDecompTransformationTest


class DecompTransformationGaussianTest(IDecompTransformationTest):
    
    __test__ = True

    def get_transformators(self) -> dict:
        return {
            "Base": DecompTransformationGaussian(),
            "Alter An":DecompTransformationGaussian(alter_approximation_coeffs=True),
            "1-mean": DecompTransformationGaussian(mean=1),
            "wider": DecompTransformationGaussian(min_noise_perc=0.001, max_noise_perc=1.1),
        }