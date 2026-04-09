from copy import deepcopy
from pywt import wavedec, waverec, dwt_max_level

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.wt_aug_base import (
    WTAugBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal


class WTAugDWT(WTAugBase):

    def __init__(
        self,
        wavelets=None,
        max_decomposition_level: int = 3,
        transformations=None,
        mode="symmetric",
    ) -> None:
        super().__init__(wavelets, max_decomposition_level, transformations)
        self.mode = mode

    def _wt_trans(self, raw_signal: RawSignal, wavelet, level: int) -> list:
        np_sig = raw_signal.to_numpy()
        max_level = dwt_max_level(len(np_sig), wavelet)
        t_level = min(level, max_level)
        decomp_list = wavedec(
            np_sig, wavelet=wavelet, level=t_level, axis=0, mode=self.mode
        )
        return decomp_list

    def _wt_itrans(self, raw_signal: RawSignal, wavelet, decomps: list) -> RawSignal:
        new_signal = deepcopy(raw_signal)
        rec_sig = waverec(coeffs=decomps, wavelet=wavelet, axis=0, mode=self.mode)
        new_signal.signal = rec_sig
        return new_signal
