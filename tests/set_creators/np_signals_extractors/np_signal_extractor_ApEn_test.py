from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ApEn import (
    NpSignalExtractorApEn,
)
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import (
    NpSignalExtractorTest,
)


class NpSignalExtractorApEnTest(NpSignalExtractorTest):

    __test__ = True

    def get_extractors(self):
        return [NpSignalExtractorApEn()]
