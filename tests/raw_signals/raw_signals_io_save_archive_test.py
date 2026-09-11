
import os
import shutil
import tempfile
import unittest
import zipfile

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals_io import (
    save_signals_to_archive,
    read_signals_from_archive,
)


class SaveSignalsToArchiveTest(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _make_signals(self, n=3, rows=10, cols=3, sample_rate=1000, channel_names=None):
        signals = RawSignals(sample_rate=sample_rate)
        for i in range(n):
            sig = np.random.random((rows, cols)).astype(np.float64)
            label = "class_{}".format(i % 2)
            signals.append(RawSignal(signal=sig, object_class=label,
                                     channel_names=channel_names, timestamp=i * 10,
                                     sample_rate=sample_rate))
        return signals

    def _archive_path(self, name="test_data.zip"):
        return os.path.join(self.test_dir, name)

    def test_creates_zip_file(self):
        signals = self._make_signals()
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(zipfile.is_zipfile(path))

    def test_archive_contains_sample_rate(self):
        signals = self._make_signals(sample_rate=2000)
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            sr_files = [n for n in names if n.endswith("sample_rate.txt")]
            self.assertEqual(len(sr_files), 1)
            content = z.read(sr_files[0]).decode("utf-8").strip()
            self.assertEqual(content, "2000")

    def test_archive_contains_channel_names(self):
        signals = self._make_signals(channel_names=["A", "B", "C"])
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            cn_files = [n for n in names if n.endswith("channel_names.txt")]
            self.assertEqual(len(cn_files), 1)
            lines = [l.strip() for l in z.read(cn_files[0]).decode("utf-8").strip().splitlines()]
            self.assertEqual(lines, ["A", "B", "C"])

    def test_archive_contains_csv_and_dat(self):
        signals = self._make_signals(n=4)
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            csv_files = [n for n in names if n.endswith(".csv")]
            dat_files = [n for n in names if n.endswith(".dat")]
            self.assertEqual(len(csv_files), 4)
            self.assertEqual(len(dat_files), 4)

    def test_roundtrip(self):
        ch_names = ["EMG1", "EMG2", "EMG3"]
        signals = self._make_signals(n=4, rows=20, cols=3, sample_rate=500,
                                     channel_names=ch_names)
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        loaded = read_signals_from_archive(path)
        accepted = loaded["accepted"]
        self.assertEqual(len(accepted), 4)
        for sig in accepted:
            self.assertEqual(sig.channel_names, tuple(ch_names))
            self.assertEqual(sig.signal.shape, (20, 3))

    def test_roundtrip_sample_rate(self):
        signals = self._make_signals(sample_rate=2000)
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        loaded = read_signals_from_archive(path)
        accepted = loaded["accepted"]
        self.assertEqual(accepted.get_sample_rate(), 2000)

    def test_roundtrip_values(self):
        signals = RawSignals(sample_rate=1000)
        data = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
        signals.append(RawSignal(signal=data, object_class="cls",
                                 channel_names=["A", "B"], timestamp=0))
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        loaded = read_signals_from_archive(path)
        accepted = loaded["accepted"]
        self.assertEqual(len(accepted), 1)
        np.testing.assert_array_almost_equal(accepted[0].signal, data, decimal=4)

    def test_roundtrip_labels(self):
        signals = self._make_signals(n=5)
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        loaded = read_signals_from_archive(path)
        accepted = loaded["accepted"]
        original_labels = sorted(signals.get_labels())
        loaded_labels = sorted(accepted.get_labels())
        self.assertEqual(list(original_labels), list(loaded_labels))

    def test_empty_signals(self):
        signals = RawSignals(sample_rate=1000)
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        self.assertTrue(zipfile.is_zipfile(path))
        with zipfile.ZipFile(path, "r") as z:
            cn_files = [n for n in z.namelist() if n.endswith("channel_names.txt")]
            self.assertEqual(len(cn_files), 0)

    def test_no_rejected(self):
        signals = self._make_signals()
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        loaded = read_signals_from_archive(path)
        self.assertIsNone(loaded["rejected"])


if __name__ == "__main__":
    unittest.main()
