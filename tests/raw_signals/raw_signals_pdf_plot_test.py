import tempfile
import unittest
import os

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import RawSignalsCreatorSines
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals_pdf_plot import raw_signals_pdf_plot

class RawSignalsPdfPlotTest(unittest.TestCase):
    
    def test_plotting(self):
        creator = RawSignalsCreatorSines(set_size=10,noise_factor=1.0)
        raw_signals = creator.get_set()
        num_signals = len(raw_signals)

        output_directory = tempfile.mkdtemp()

        raw_signals_pdf_plot(out_dir_path=output_directory, raw_signals=raw_signals)

        file_list = [f for f in os.listdir(output_directory) if f.endswith(".pdf") ]
        self.assertTrue( num_signals == len(file_list), "Number of generated files." )



if __name__ == '__main__':
    unittest.main()