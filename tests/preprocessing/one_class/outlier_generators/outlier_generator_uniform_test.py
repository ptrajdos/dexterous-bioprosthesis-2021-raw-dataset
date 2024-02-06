
from dexterous_bioprosthesis_2021_raw_datasets.preprocessing.one_class.outlier_generators.outlier_generator_uniform import OutlierGeneratorUniform
from tests.preprocessing.one_class.outlier_generators.outlier_generator_test import OutlierGeneratorTest


class OutlierGeneratorUniformTest(OutlierGeneratorTest):
    __test__ = True

    def get_generators(self):
        return [
            OutlierGeneratorUniform()
        ]