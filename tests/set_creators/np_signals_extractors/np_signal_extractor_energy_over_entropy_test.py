from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_energy_over_entropy import (
    NpSignalExtractorEnergyOverEntropy,
)
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import (
    NpSignalExtractorTest,
)


class NpSignalExtractorEnergyOverEntropyTest(NpSignalExtractorTest):

    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorEnergyOverEntropy(),
            NpSignalExtractorEnergyOverEntropy(
                sanitize_output=True, check_input=True, check_output=True
            ),
        ]
