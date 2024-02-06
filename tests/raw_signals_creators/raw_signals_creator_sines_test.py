
import unittest

from tests.raw_signals_creators.raw_signals_creator_test import RawSignalsCreatorTest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import RawSignalsCreatorSines

class RawSignalsCreatorSinesTest(RawSignalsCreatorTest):

    __test__ = True

    def get_creator(self):
        return RawSignalsCreatorSines()

if __name__ == '__main__':
    unittest.main()