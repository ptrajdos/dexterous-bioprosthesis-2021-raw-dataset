from tests.data_augumentation.raw_signals_augumenter_test import RawSignalsAugumenterTest
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_white_noise import  RawSignalsAugumenterWhiteNoise


class RawSignalsAugumenterWhiteNoiseTest(RawSignalsAugumenterTest):


    __test__ = True

    def get_augumenter(self):
        return RawSignalsAugumenterWhiteNoise()