from tests.data_augumentation.raw_signals_augumenter_test import RawSignalsAugumenterTest
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_pitch_shift import RawSignalsAugumenterPitchShift

try:
    import audiomentations
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False

class RawSignalsAugumenterPitchShiftTest(RawSignalsAugumenterTest):



    __test__ = HAS_AUDIOMENTATIONS

    def get_augumenter(self):
        return RawSignalsAugumenterPitchShift()