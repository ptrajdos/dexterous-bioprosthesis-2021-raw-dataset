from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_time_jiter import (
    RawSignalsAugumenterTimeJiter,
)
from tests.data_augumentation.raw_signals_augumenter_test import (
    RawSignalsAugumenterTest,
)


class RawSignalsAugumenterTimeJiterTest(RawSignalsAugumenterTest):

    __test__ = True

    def get_augumenter(self):
        return RawSignalsAugumenterTimeJiter()
