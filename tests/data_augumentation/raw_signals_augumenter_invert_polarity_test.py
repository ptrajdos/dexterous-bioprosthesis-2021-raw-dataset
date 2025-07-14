from tests.data_augumentation.raw_signals_augumenter_test import RawSignalsAugumenterTest
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_invert_polarity import RawSignalsAugumenterInvertPolarity


class RawSignalsAugumenterInvertPolarityTest(RawSignalsAugumenterTest):


    __test__ = True

    def get_augumenter(self):
        return RawSignalsAugumenterInvertPolarity()