
import pywt

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_wt_abstract import SetCreatorWTAbstract

class SetCreatorDWT(SetCreatorWTAbstract):
    
    def _decompose_signal(self, signal, fs=1000):
        wavelet = pywt.Wavelet(self.wavelet_name)
        coeffs = pywt.wavedec(signal, wavelet=wavelet,axis=0, level=self.num_levels)
        coeffs_with_fs = []
        for i, c in enumerate(coeffs):
            j = self.num_levels - i if i > 0 else self.num_levels
            f_s_j = fs / (2 ** j)
            coeffs_with_fs.append((c, f_s_j))
        return coeffs_with_fs