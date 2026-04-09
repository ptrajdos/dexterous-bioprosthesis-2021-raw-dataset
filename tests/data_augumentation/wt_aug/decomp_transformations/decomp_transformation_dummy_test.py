from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_dummy import DecompTransformationDummy
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from tests.data_augumentation.wt_aug.decomp_transformations.idecomp_transformation_test import IDecompTransformationTest


class DecompTransformationDummyTest(IDecompTransformationTest):
    
    __test__ = True

    def get_transformators(self) -> dict:
        return {
            "Base": DecompTransformationDummy(),
        }
    
    def _compare_signals(self, orig_raw: RawSignals, trans_raw: RawSignals):
        pass