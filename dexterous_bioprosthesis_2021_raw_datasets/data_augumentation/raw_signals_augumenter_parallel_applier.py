from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_invert_polarity import (
    RawSignalsAugumenterInvertPolarity,
)
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_white_noise import (
    RawSignalsAugumenterWhiteNoise,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsAugumenterParallelApplier(RawSignalsAugumenter):

    def __init__(self, augumenter_list=None, append_original=True) -> None:
        super().__init__()

        self.augumenter_list = augumenter_list
        self.append_original = append_original

    def _prepare_effective_augumenter_list(self):
        if self.augumenter_list is None or len(self.augumenter_list) == 0:
            self._augumenter_list = [
                RawSignalsAugumenterInvertPolarity(append_original=False),
                RawSignalsAugumenterWhiteNoise(
                    noise_perc_min=0.2, n_repeats=2, append_original=False
                ),
            ]
        else:
            self._augumenter_list = self.augumenter_list

    def _check_fitted(self):
        if not hasattr(self, "_augumenter_list"):
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def fit(self, raw_signals: RawSignals):
        """
        Intentionally does nothing
        """
        self._prepare_effective_augumenter_list()

        for aug in self._augumenter_list:
            aug.fit(raw_signals)
        return self

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        self._check_fitted()

        new_signals = raw_signals.initialize_empty()

        for aug in self._augumenter_list:
            new_signals += aug.transform(raw_signals)

        if self.append_original:
            new_signals += raw_signals

        return new_signals

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        self.fit(raw_signals)
        return self.transform(raw_signals)
