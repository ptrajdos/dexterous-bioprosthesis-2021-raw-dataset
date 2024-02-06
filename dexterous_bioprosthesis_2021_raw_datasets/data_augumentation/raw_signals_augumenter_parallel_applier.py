from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import RawSignalsAugumenter
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from copy import deepcopy

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_invert_polarity import RawSignalsAugumenterInvertPolarity
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_white_noise import RawSignalsAugumenterWhiteNoise
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_gain_channel import RawSignalsAugumenterGainChannel
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter_clipping_distortion import RawSignalsAugumenterClippingDistortion

default_augumenter_list = [
    RawSignalsAugumenterInvertPolarity(append_original=False),
    RawSignalsAugumenterWhiteNoise(noise_perc_min=0.2, n_repeats=2, append_original=False),
    RawSignalsAugumenterGainChannel(n_repeats=3, gain_perc_min=0.1, append_original=False), 
    RawSignalsAugumenterClippingDistortion(n_repeats=2, append_original= False)
]

class RawSignalsAugumenterParallelApplier(RawSignalsAugumenter):

    def __init__(self, augumenter_list = default_augumenter_list) -> None:
        super().__init__()

        self.augumenter_list = augumenter_list

    def fit(self, raw_signals: RawSignals):
        """
        Intentionally does nothing
        """
        for aug in self.augumenter_list:
            aug.fit(raw_signals)
        return self

    def transform(self, raw_signals: RawSignals) -> RawSignals:

        new_signals = RawSignals()

        for aug in self.augumenter_list:
            new_signals += aug.transform(raw_signals)

        new_signals += raw_signals

        return new_signals

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        self.fit(raw_signals)
        return self.transform(raw_signals)
        