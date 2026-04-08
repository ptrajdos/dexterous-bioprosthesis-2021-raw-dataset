from copy import deepcopy

import numpy as np
from sklearn.exceptions import NotFittedError

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsAugumenterDummy(RawSignalsAugumenter):

    def __init__(self) -> None:
        super().__init__()

    def _check_fittesness(self):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def fit(self, raw_signals: RawSignals):
        self._is_fitted = True
        return self

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        self._check_fittesness()
        return deepcopy(raw_signals)

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        self.fit(raw_signals)
        return self.transform(raw_signals)

    def sample(self, raw_signals: RawSignals, n_samples: int=1) -> RawSignals:
        self._check_fittesness()
        n_signals = len(raw_signals)

        replace = n_samples > n_signals
        indices = np.random.choice(n_signals, size=n_samples, replace=replace)
        new_signals = raw_signals.initialize_empty()
        for idx in indices:
            new_signals += [deepcopy(raw_signals[idx])]

        return new_signals