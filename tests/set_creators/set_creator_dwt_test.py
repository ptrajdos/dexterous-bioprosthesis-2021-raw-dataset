import unittest

import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ar import NpSignalExtractorAr
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mav import NpSignalExtractorMav
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ssc import NpSignalExtractorSsc
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_dwt import SetCreatorDWT
from tests.set_creators.set_creator_test import SetCreatorTest

import warnings

from tests.testing_tools import generate_sample_data


class SetCreatorDWTTest(SetCreatorTest):
    
    __test__ = True
    def get_creators(self) :
        extractors = [
            SetCreatorDWT(extractors=[
                NpSignalExtractorMav(),
                NpSignalExtractorSsc(),
                NpSignalExtractorAr(),
            ])
            ]
        return extractors
    
    def generate_sample_data(self, dtype=np.double):
        return generate_sample_data(samples_number=1000, dtype=dtype)


if __name__ == '__main__':

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        unittest.main()