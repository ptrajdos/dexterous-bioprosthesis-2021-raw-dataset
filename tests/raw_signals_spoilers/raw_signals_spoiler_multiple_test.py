from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_damper import RawSignalsSpoilerDamper
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_gauss import RawSignalsSpoilerGauss
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_multiple import RawSignalsSpoilerMultiple
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_sine import RawSignalsSpoilerSine
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.relabelers.relabeler_constant import RelabelerConstant
from tests.raw_signals_spoilers.raw_signals_spoiler_interface_test import RawSignalsSpoilerInterfaceTest



class RawSignalsSpoilerMultipleTest(RawSignalsSpoilerInterfaceTest):

    __test__ = True

    def get_spoilers(self):
        return [
            RawSignalsSpoilerMultiple(),
            RawSignalsSpoilerMultiple(spoiled_fraction=0.5),
            RawSignalsSpoilerMultiple(spoiled_fraction=1.5),

            RawSignalsSpoilerMultiple(spoilers=[
                RawSignalsSpoilerGauss(),
                RawSignalsSpoilerSine(),    
                RawSignalsSpoilerDamper()
            ],
            spoilers_weights=[0.1,0.5,0.4],
            spoiler_relabalers=[RelabelerConstant("A"),RelabelerConstant("B"),RelabelerConstant("C")]
            ),

        ]
