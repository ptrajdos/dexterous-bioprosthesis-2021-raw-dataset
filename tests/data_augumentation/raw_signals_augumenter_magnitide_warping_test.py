from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_magnitude_warping import (
    RawSignalsAugumenterMagnitudeWarping,
)
from tests.data_augumentation.raw_signals_augumenter_test import (
    RawSignalsAugumenterTest,
)


class RawSignalsAugumenterMagnitudeWarpingTest(RawSignalsAugumenterTest):

    __test__ = True

    def get_augumenter(self):
        return RawSignalsAugumenterMagnitudeWarping()
