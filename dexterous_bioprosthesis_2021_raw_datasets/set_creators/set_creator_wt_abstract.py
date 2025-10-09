
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import SetCreator
from sklearn.exceptions import NotFittedError
import numpy as np 
import pywt
import abc

class SetCreatorWTAbstract(SetCreator):
    
    def __init__(self, wavelet_name="db1", num_levels=2, extractors=[]) -> None:
        super().__init__()
        self.wavelet_name = wavelet_name
        self.num_levels = num_levels
        self.extractors = extractors

        self._num_attribs = None
        self.n_channels = None
        self.channel_selected_attribs = None # List containing number of attributes for each channel


    def fit(self, raw_signals: RawSignals, y=None):
        
        self.n_channels = raw_signals[0].to_numpy().shape[1]
        n_extractors = len(self.extractors)
        n_levels  = self.num_levels+1

        self.channel_selected_attribs = []
        for ch_id in range(self.n_channels):
            self.channel_selected_attribs.append([])

        offset = 0
        for extr_id in range(n_extractors):
            n_attribs_per_channel =self.extractors[extr_id].attribs_per_column()
            for level_id in range(n_levels):
                for ch_id in range(self.n_channels):
                    tmp_attrib_idxs = [ offset + i for i in range(n_attribs_per_channel)]
                    self.channel_selected_attribs[ch_id] += tmp_attrib_idxs
                    offset += n_attribs_per_channel

        self._num_attribs = offset

        return self
    
    @abc.abstractmethod
    def _decompose_signal(self, signal):
        pass

    def transform(self, raw_signals: RawSignals):
        
        if self.get_channel_attribs_indices() is None:
            raise NotFittedError("SetCreator has not been fitted.")
        
        
        n_signals = len(raw_signals)
        extracted_attribs = np.zeros( (n_signals, self._num_attribs))
        labels = []
        timestamps = []

        for raw_signal_id,  raw_signal in enumerate(raw_signals):
            
            signal = raw_signal.to_numpy()
            labels.append(raw_signal.get_label())
            timestamps.append(raw_signal.get_timestamp()) 
            decomposeds = self._decompose_signal(signal= signal)
            offset = 0
            for extractor_id, extractor in enumerate(self.extractors):
                for decomposed_level in decomposeds:
                    extracted = extractor.fit_transform(decomposed_level)
                    n_extracted = extracted.shape[0]
                    extracted_attribs[raw_signal_id, offset:(offset+n_extracted)] = extracted
                    offset += n_extracted
                    
        extracted_attribs = extracted_attribs.astype(raw_signals[0].to_numpy().dtype)
        labels = np.asanyarray(labels)
        timestamps = np.asanyarray(timestamps)
        return extracted_attribs, labels, timestamps
    
    def fit_transform(self, raw_signals: RawSignals, y=None):
        return self.fit(raw_signals).transform(raw_signals)
    

    def get_channel_attribs_indices(self):
        return self.channel_selected_attribs