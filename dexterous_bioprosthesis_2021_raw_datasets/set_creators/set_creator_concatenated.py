from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import SetCreator
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_feature_extractor import SetCreatorFeatureExtractor

import pandas as pd
import numpy as np


class SetCreatorConcatenated(SetCreator):

    def __init__(self, creators) -> None:
        super().__init__()

        self.creators = creators
        self.channel_selected_attribs = None # List containing number of attributes for each channel

    def fit(self, raw_signals: RawSignals, y=None):
        
        for creator in self.creators:
            creator.fit(raw_signals)

        is_creators_channel_attribs = [ True if creator.get_channel_attribs_indices() is not None else False for creator in self.creators ]
        if all(is_creators_channel_attribs):
            self.channel_selected_attribs = None
            
            for creator in self.creators:
                creator_channel_attribs = creator.get_channel_attribs_indices()
                if self.channel_selected_attribs is None:
                    self.channel_selected_attribs = creator_channel_attribs
                else:
                    offset = self._find_offset(self.channel_selected_attribs)
                    for channel_idx, attrib_indices in enumerate(creator_channel_attribs):
                        self.channel_selected_attribs[channel_idx] += list(offset +  np.asanyarray(attrib_indices))

        return self
    
    def _find_offset(self, channel_selected_attribs):
        curr_max = 0
        for li in channel_selected_attribs:
            tmp_max = np.max(li)
            if tmp_max > curr_max:
                curr_max = tmp_max
        return curr_max + 1

    def transform(self, raw_signals: RawSignals):
        X_es = []
        y_f = None
        t_f = None

        for creator in self.creators:
            X, y, t = creator.transform(raw_signals)

            if y_f is None:
                y_f = y
                t_f = t

            if isinstance(X, pd.DataFrame ):
                X = pd.DataFrame.to_numpy(X)

            X_es.append(X)
        
        Xf = np.concatenate(X_es, axis=1)
        
        return Xf, y_f, t_f
    

    def fit_transform(self, raw_signals: RawSignals, y=None):
        self.fit(raw_signals)
        return self.transform(raw_signals)
    
    def get_channel_attribs_indices(self):
        return self.channel_selected_attribs