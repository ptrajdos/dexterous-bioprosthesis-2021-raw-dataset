import logging
from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signal_augumenter_base import (
    RawSignalsAugumenterBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals

try:
    from audiomentations import PitchShift
except ImportError:
    logging.warning(
        "audiomentations is not installed. RawSignalsAugumenterPitchShift will not work."
    )


class RawSignalsAugumenterPitchShift(RawSignalsAugumenterBase):

    def __init__(
        self,
        min_semitones=-4,
        max_semitones=4,
        n_repeats: int = 2,
        append_original=True,
        n_jobs=None,
    ) -> None:
        super().__init__(n_jobs=n_jobs, append_original=append_original)

        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.n_repeats = n_repeats

    def fit(self, raw_signals: RawSignals):
        """
        Intentionally does nothing
        """
        return self

    def _sig_augument(self, raw_signal: RawSignal):
        sample_rate = raw_signal.sample_rate
        sig_list = []
        transformer = PitchShift(
            p=1.0, min_semitones=self.min_semitones, max_semitones=self.max_semitones
        )

        for _ in range(self.n_repeats):
            new_signal = deepcopy(raw_signal)
            np_sig = new_signal.signal

            for ch_id in range(np_sig.shape[1]):

                ch_sig = np_sig[:, ch_id]
                # FIXME  Woraround for  nummba issue with different dtypes
                # FIXME NotImplementedError: Failed in nopython mode pipeline (step: native lowering)
                np_sig[:, ch_id] = transformer(
                    ch_sig.astype(np.double), sample_rate=sample_rate
                ).astype(np_sig.dtype)
            sig_list.append(new_signal)

        return sig_list
