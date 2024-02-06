
import unittest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_channel_idx import RawSignalsFilterChannelIdx
from tests.raw_signals_filters.raw_signals_filter_test import RawSignalsFilterTest


class RawSignalsFilterChannelIdxTest(RawSignalsFilterTest):
    __test__ = True

    def get_filters(self):
        indices_list = [0,1,2]
        return [RawSignalsFilterChannelIdx(indices_list=indices_list)]

    def test_num_sel(self):
        n_base_columns = 10
        raw_signals = self.generate_sample_data(column_number=n_base_columns)

        sel_col_indices = [0,3,5,7,9]

        filter  = RawSignalsFilterChannelIdx(indices_list=sel_col_indices)

        try:
            filtered_signals = filter.fit_transform(raw_signals)
            
            self.assertTrue(len(filtered_signals) == len(raw_signals), "The number of objects should not change")
            self.assertTrue(filtered_signals.signal_n_cols == len(sel_col_indices), "Wrong number of selected columns" )

        except Exception as ex:
            self.fail("An exception has been caught {}".format(ex))

        
if __name__ == '__main__':
    unittest.main()