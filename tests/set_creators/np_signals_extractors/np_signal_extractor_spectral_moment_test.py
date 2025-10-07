from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_spectral_moment import (
    NpSignalExtractorSpectralMoment,
)
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import (
    NpSignalExtractorTest,
)


class NpSignalExtractorSpectralMomentTest(NpSignalExtractorTest):

    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorSpectralMoment(),
            NpSignalExtractorSpectralMoment(
                sanitize_output=True, check_input=True, check_output=True
            ),
            NpSignalExtractorSpectralMoment(order=2),
            NpSignalExtractorSpectralMoment(order=3, centered=False)
        ]
