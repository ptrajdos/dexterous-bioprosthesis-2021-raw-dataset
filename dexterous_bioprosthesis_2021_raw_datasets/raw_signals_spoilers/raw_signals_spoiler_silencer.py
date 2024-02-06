

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler import RawSignalsSpoiler
from copy import deepcopy

class RawSignalsSpoilerSilencer(RawSignalsSpoiler):

    def __init__(self, channels_spoiled_frac=0.1, snr=1) -> None:
        super().__init__(channels_spoiled_frac, snr)

    def fit(self,raw_signals:RawSignals):
        # Does nothing
        return self

    def transform(self, raw_signals:RawSignals):
        copied_signals = deepcopy(raw_signals)

        for signal in copied_signals:
            selected_channels_idxs = self._random_channel_selection(signal)
            np_repr = signal.to_numpy()
            np_repr[:,selected_channels_idxs] = 0

        return copied_signals
        