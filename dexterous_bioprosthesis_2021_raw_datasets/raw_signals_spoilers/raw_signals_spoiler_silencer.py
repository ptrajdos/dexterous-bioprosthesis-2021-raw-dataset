from copy import deepcopy

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler import (
    RawSignalsSpoiler,
)


class RawSignalsSpoilerSilencer(RawSignalsSpoiler):

    def __init__(
        self,
        channels_spoiled_frac=0.1,
        snr=1,
        random_state=10,
    ) -> None:
        super().__init__(channels_spoiled_frac, snr, random_state)

    def transform(self, raw_signals: RawSignals):
        self._check_is_fitted()
        copied_signals = deepcopy(raw_signals)

        for signal in copied_signals:
            selected_channels_idxs = self._random_channel_selection(signal)
            np_repr = signal.to_numpy()
            np_repr[:, selected_channels_idxs] = 0

        return copied_signals
