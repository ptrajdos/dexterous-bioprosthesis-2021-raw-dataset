from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.sampler_mixin import SamplerMixin
from tests.data_augumentation.raw_signals_augumenter_test import (
    RawSignalsAugumenterTest,
)

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_standarizer import (
    RawSignalsFilterAllStandarizer,
)

class SamplerMixinStandarizer(RawSignalsFilterAllStandarizer, SamplerMixin):
    pass

class SamplerMixinTest(RawSignalsAugumenterTest):

    __test__ = True

    def get_augumenter(self): # type: ignore
        return SamplerMixinStandarizer()
