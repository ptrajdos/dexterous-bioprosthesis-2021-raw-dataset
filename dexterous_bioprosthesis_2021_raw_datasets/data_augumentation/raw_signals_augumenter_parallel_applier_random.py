import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_parallel_applier import (
    RawSignalsAugumenterParallelApplier,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsAugumenterParallelApplierRandom(RawSignalsAugumenterParallelApplier):

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        self._check_fitted()
        new_signals = raw_signals.initialize_empty()

        n_augs = len(self._augumenter_list)
        assigs = np.random.randint(0, n_augs, len(raw_signals), dtype=np.uint)

        for aug_idx, aug in enumerate(self._augumenter_list):
            sig_selected = assigs == aug_idx
            new_signals += aug.transform(raw_signals[sig_selected])

        if self.append_original:
            new_signals += raw_signals

        return new_signals
