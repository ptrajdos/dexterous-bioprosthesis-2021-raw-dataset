import unittest

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ar import (
    NpSignalExtractorAr,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mav import (
    NpSignalExtractorMav,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ssc import (
    NpSignalExtractorSsc,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_swt import (
    SetCreatorSWT,
)
from tests.set_creators.set_creator_test import SetCreatorTest

import warnings

from tests.testing_tools import generate_sample_data


class SetCreatorSWTTest(SetCreatorTest):

    __test__ = True

    def get_creators(self):
        extractors = [
            SetCreatorSWT(
                extractors=[
                    NpSignalExtractorMav(),
                    NpSignalExtractorSsc(),
                ]
            ),
            SetCreatorSWT(
                extractors=[
                    NpSignalExtractorMav(),
                    NpSignalExtractorSsc(),
                ],
                num_levels=3,
            ),
        ]
        return extractors

    def get_sample_data_parameters(self):
        return [
            (10, 3, 1200, [0, 1]),
            (11, 2, 1203, [0, 1]),
            (10, 1, 1200, [0, 1, 2]),
            (10, 5, 1205, [0, 1, 2, 3]),
        ]
    
    def get_default_sample_number(self):
        return 1000

if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        unittest.main()
