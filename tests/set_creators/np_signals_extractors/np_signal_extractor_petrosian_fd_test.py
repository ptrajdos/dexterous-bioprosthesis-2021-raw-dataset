import unittest
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_petrosian_fd import (
    NpSignalExtractorPetrosianFD,
)
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import (
    NpSignalExtractorTest,
)


class NpSignalExtractorPetrosianFDTest(NpSignalExtractorTest):

    __test__ = True

    def get_extractors(self):
        return [NpSignalExtractorPetrosianFD()]
