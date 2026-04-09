from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_gaussian import DecompTransformationGaussian
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.wt_aug_dwt import WTAugDWT
from tests.data_augumentation.raw_signals_augumenter_test import RawSignalsAugumenterTest


class WTAugDWTTest(RawSignalsAugumenterTest):


    __test__ = True

    def get_augumenter(self):
        return WTAugDWT(transformations=[DecompTransformationGaussian()])