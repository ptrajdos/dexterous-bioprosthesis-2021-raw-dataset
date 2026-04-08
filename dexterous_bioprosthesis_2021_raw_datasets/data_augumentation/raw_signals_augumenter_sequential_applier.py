from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_parallel_applier import RawSignalsAugumenterParallelApplier
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from copy import deepcopy




class RawSignalsAugumenterSequentialApplier(RawSignalsAugumenterParallelApplier):

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        self._check_fitted()

        tmp_signals = deepcopy(raw_signals)

        for aug in self._augumenter_list:
            tmp_signals += aug.transform(tmp_signals)

        return tmp_signals
