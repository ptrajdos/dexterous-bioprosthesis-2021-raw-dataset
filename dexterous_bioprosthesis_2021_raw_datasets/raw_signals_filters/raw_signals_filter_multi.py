from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_pass import RawSignalsFilterAllPass


class RawSignalsFilterMulti(RawSignalsFilter):
    def __init__(self, filter_list=[RawSignalsFilterAllPass()]) -> None:
        super().__init__()

        self.filter_list = filter_list

    def fit(self, raw_signals: RawSignals) -> None:
        #Does nothing. Fit is lazy
        return super().fit(raw_signals)
    
    def transform(self,raw_signals: RawSignals)->RawSignals:
        
        pre_signals = raw_signals
        post_signals = None
        for filter in self.filter_list:
            #lazy fitting. Depends on filter order
            post_signals = filter.fit_transform(pre_signals)
            pre_signals = post_signals

        return post_signals


