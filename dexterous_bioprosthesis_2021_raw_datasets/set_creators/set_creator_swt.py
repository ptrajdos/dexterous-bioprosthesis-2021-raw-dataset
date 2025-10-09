import pywt

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_wt_abstract import (
    SetCreatorWTAbstract,
)
import itertools

class SetCreatorSWT(SetCreatorWTAbstract):

    def __init__(
        self, wavelet_name="db1", num_levels=2, extractors=[], norm: bool = False
    ) -> None:
        super().__init__(
            wavelet_name=wavelet_name, num_levels=num_levels, extractors=extractors
        )
        self._norm = norm

    def _decompose_signal(self, signal, fs=1000):
        wavelet = pywt.Wavelet(self.wavelet_name)
        rem = len(signal) % 2 ** self.num_levels
        if rem > 0:
            tmp_dat = signal[:-rem]
        else:
            tmp_dat = signal
        
        coeffs =  pywt.swt(
            tmp_dat,
            wavelet=wavelet,
            axis=0,
            level=self.num_levels,
            trim_approx=True,
            norm=self._norm,
        )

        freqs = itertools.repeat(fs, self.num_levels + 1)
        coeffs_with_fs = list(zip(coeffs, freqs))

        return coeffs_with_fs

    
