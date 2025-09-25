
import pywt

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_wt_abstract import SetCreatorWTAbstract

class SetCreatorDWT(SetCreatorWTAbstract):
    
    def _decompose_signal(self, signal):
        wavelet = pywt.Wavelet(self.wavelet_name)
        return pywt.wavedec(signal, wavelet=wavelet,axis=0, level=self.num_levels)