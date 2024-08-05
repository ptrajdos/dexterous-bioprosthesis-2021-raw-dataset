import unittest
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ar import NpSignalExtractorAr
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mav import NpSignalExtractorMav
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ssc import NpSignalExtractorSsc
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_functions import SetCreatorFunctions
from tests.set_creators.set_creator_test import SetCreatorTest

import warnings

from tests.testing_tools import generate_sample_data


class SetCreatorFunctionsTest(SetCreatorTest):
    
    __test__ = True
    def get_creators(self) :
        extractors = [
            SetCreatorFunctions(extractors=[
                NpSignalExtractorMav(),
                NpSignalExtractorSsc(),
            ])
            ]
        return extractors
    
    def generate_sample_data(self):
        return generate_sample_data(samples_number=1000)
