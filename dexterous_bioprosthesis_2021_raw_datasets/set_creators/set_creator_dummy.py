from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator import SetCreator


class SetCreatorDummy(SetCreator):
    
    def fit(self, raw_signals: RawSignals, y=None):
        return super().fit(raw_signals)
    
    def transform(self, raw_signals: RawSignals):

        X = raw_signals
        y = raw_signals.get_labels()
        t = raw_signals.get_timestamps()
        return X,y,t
    
    def fit_transform(self, raw_signals: RawSignals, y=None):
        self.fit(raw_signals)

        return self.transform(raw_signals) 
    
    def get_channel_attribs_indices(self):
        return None