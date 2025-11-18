from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_parallel_applier_random import (
    RawSignalsAugumenterParallelApplierRandom,
)
from tests.data_augumentation.raw_signals_augumenter_test import (
    RawSignalsAugumenterTest,
)

class RawSignalsAugumenterParallelApplierTest(RawSignalsAugumenterTest):

    __test__ = True

    def get_augumenter(self):
        return RawSignalsAugumenterParallelApplierRandom()
