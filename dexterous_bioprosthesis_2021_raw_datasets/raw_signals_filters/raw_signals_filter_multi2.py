from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import RawSignalsFilter
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_all_pass import RawSignalsFilterAllPass


class RawSignalsFilterMulti2(RawSignalsFilter):
    def __init__(self, filter_list=[RawSignalsFilterAllPass()]) -> None:
        super().__init__()

        self.filter_list = filter_list

    def fit(self, raw_signals: RawSignals) -> None:
        #Does nothing. Fit is lazy
        return super().fit(raw_signals)
    
    def transform(self,raw_signals: RawSignals)->RawSignals:
        
        
        post_signals = None
        for filter in self.filter_list:
            #lazy fitting. Depends on filter order
            tmp_signals = filter.fit_transform(raw_signals)
            if post_signals is None:
                post_signals = tmp_signals
            else:
                post_signals += tmp_signals

        return post_signals


