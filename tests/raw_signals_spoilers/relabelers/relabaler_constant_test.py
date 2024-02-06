from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.relabelers.relabeler_constant import RelabelerConstant
from tests.raw_signals_spoilers.relabelers.relabeler_test import RelabelerTest


class RelabelerConstantTest(RelabelerTest):

    __test__ = True

    def get_relabelers(self):

        return [
            RelabelerConstant(),
            RelabelerConstant(new_label="Z")
        ]