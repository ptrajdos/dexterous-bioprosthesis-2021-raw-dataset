import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ar import NpSignalExtractorAr
from tests.set_creators.np_signals_extractors.np_signal_extractor_test import NpSignalExtractorTest


class NpSignalExtractorArTest(NpSignalExtractorTest):


    __test__ = True

    def get_extractors(self):
        return [
            NpSignalExtractorAr(),
            NpSignalExtractorAr(check_input=True),
        ]

    def test_attribute_names_values(self):
        self.assertEqual(
            NpSignalExtractorAr(lags=3).get_attribute_names(),
            ["ar_const", "ar_lag1", "ar_lag2", "ar_lag3"],
        )
