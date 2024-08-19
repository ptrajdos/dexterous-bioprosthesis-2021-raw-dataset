import unittest
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
import tests.settings as settings
import os
import tempfile
import shutil

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals_io import read_signals_from_dirs, save_signals_to_dirs

class RawSignalsIOTest(unittest.TestCase):

    def test_reading(self):
        data_path = os.path.join(settings.DATAPATH, "Andrzej_19_10_2022")

        try:
            signals = read_signals_from_dirs(data_path)
            self.assertIn("accepted", signals)
            self.assertIn("rejected", signals)

            self.assertTrue( isinstance(signals["accepted"], RawSignals) )
            self.assertTrue( isinstance(signals["rejected"], RawSignals) )

            for acc in signals["accepted"]:
                self.assertTrue( isinstance(acc, RawSignal) )

            for rej in signals["rejected"]:
                self.assertTrue( isinstance(rej, RawSignal) )

        except Exception as ex:
            self.fail("An exception has been caught during reading dataset from the file structure: " + str(ex))

    def test_reading_parallel_multiprocessing(self):
        data_path = os.path.join(settings.DATAPATH, "Andrzej_19_10_2022")

        try:
            signals = read_signals_from_dirs(data_path, parallel_options={'backend':'multiprocessing'})
            self.assertIn("accepted", signals)
            self.assertIn("rejected", signals)

            self.assertTrue( isinstance(signals["accepted"], RawSignals) )
            self.assertTrue( isinstance(signals["rejected"], RawSignals) )

            for acc in signals["accepted"]:
                self.assertTrue( isinstance(acc, RawSignal) )

            for rej in signals["rejected"]:
                self.assertTrue( isinstance(rej, RawSignal) )

        except Exception as ex:
            self.fail("An exception has been caught during reading dataset from the file structure: " + str(ex))

    def test_reading_parallel_threading(self):
        data_path = os.path.join(settings.DATAPATH, "Andrzej_19_10_2022")

        try:
            signals = read_signals_from_dirs(data_path, parallel_options={'backend':'threading'})
            self.assertIn("accepted", signals)
            self.assertIn("rejected", signals)

            self.assertTrue( isinstance(signals["accepted"], RawSignals) )
            self.assertTrue( isinstance(signals["rejected"], RawSignals) )

            for acc in signals["accepted"]:
                self.assertTrue( isinstance(acc, RawSignal) )

            for rej in signals["rejected"]:
                self.assertTrue( isinstance(rej, RawSignal) )

        except Exception as ex:
            self.fail("An exception has been caught during reading dataset from the file structure: " + str(ex))
    
    def test_writting(self):
        data_path = os.path.join(settings.DATAPATH, "Andrzej_19_10_2022")
        signals = read_signals_from_dirs(data_path)
        output_directory = tempfile.mkdtemp()
        # output_directory = os.path.join(settings.DATAPATH, output_directory)

        raw_signals = signals["accepted"]
        save_signals_to_dirs(raw_signals, output_directory)

        re_read_signals = read_signals_from_dirs(output_directory)["accepted"]

        self.assertTrue( len(re_read_signals) == len(raw_signals), "Signals differ in length")

        self.assertTrue(  set(re_read_signals.get_labels()) == set(raw_signals.get_labels()), "Different set of labels" )

        self.assertTrue( raw_signals == re_read_signals, "Datasets should have been the same!")


if __name__ == '__main__':
    unittest.main()