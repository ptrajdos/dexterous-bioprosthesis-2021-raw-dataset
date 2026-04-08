import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_parallel_applier import (
    RawSignalsAugumenterParallelApplier,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from copy import deepcopy


class RawSignalsAugumenterSequentialApplier(RawSignalsAugumenterParallelApplier):

    def _inner_transform(self, raw_signals: RawSignals) -> RawSignals:
        tmp_signals = deepcopy(raw_signals)

        for aug in self._augumenter_list:
            tmp_signals += aug.transform(tmp_signals)

        return tmp_signals

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        self._check_fitted()
        return self._inner_transform(raw_signals)

    def sample(self, raw_signals: RawSignals, n_samples: int = 1) -> RawSignals:
        self._check_fitted()

        new_sigsnals_pre = self._inner_transform(raw_signals)
        n_new_signals = len(new_sigsnals_pre)
        replace = n_samples > n_new_signals
        indices = np.random.choice(n_new_signals, size=n_samples, replace=replace)
        new_signals = new_sigsnals_pre.initialize_empty()
        for idx in indices:
            new_signals += [new_sigsnals_pre[idx]]

        return new_signals
