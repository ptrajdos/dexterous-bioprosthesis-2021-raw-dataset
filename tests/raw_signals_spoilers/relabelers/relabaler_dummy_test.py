

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.relabelers.relabeler_dummy import RelabelerDummy
from tests.raw_signals_spoilers.relabelers.relabeler_test import RelabelerTest


class RelabelerDummyTest(RelabelerTest):

    __test__ = True

    def get_relabelers(self):

        return [
            RelabelerDummy(),
        ]