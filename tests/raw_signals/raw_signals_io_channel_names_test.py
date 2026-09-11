
import os
import shutil
import tempfile
import unittest

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals_io import (
    save_signals_to_dirs,
    read_signals_from_dirs,
    save_signals_to_archive,
    read_signals_from_archive,
)


class RawSignalsIOChannelNamesTest(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _make_signals(self, channel_names, n=3, rows=10, cols=None):
        if cols is None:
            cols = len(channel_names)
        signals = RawSignals(sample_rate=1000)
        for i in range(n):
            sig = np.random.random((rows, cols)).astype(np.float64)
            signals.append(RawSignal(signal=sig, object_class="aclassA",
                                     channel_names=channel_names, timestamp=i))
        return signals

    def test_channel_names_file_created(self):
        signals = self._make_signals(["EMG1", "EMG2", "EMG3"])
        save_signals_to_dirs(signals, self.test_dir)
        cn_path = os.path.join(self.test_dir, "channel_names.txt")
        self.assertTrue(os.path.exists(cn_path))

    def test_channel_names_file_content(self):
        names = ["EMG1", "EMG2", "EMG3"]
        signals = self._make_signals(names)
        save_signals_to_dirs(signals, self.test_dir)
        cn_path = os.path.join(self.test_dir, "channel_names.txt")
        with open(cn_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        self.assertEqual(lines, names)

    def test_roundtrip_preserves_channel_names(self):
        names = ["Sensor_A", "Sensor_B", "Sensor_C"]
        signals = self._make_signals(names)
        save_signals_to_dirs(signals, self.test_dir)
        loaded = read_signals_from_dirs(self.test_dir, n_jobs=1)
        accepted = loaded["accepted"]
        self.assertTrue(len(accepted) > 0)
        for sig in accepted:
            self.assertEqual(sig.channel_names, tuple(names))

    def test_roundtrip_default_channel_names(self):
        signals = self._make_signals(None, cols=3)
        save_signals_to_dirs(signals, self.test_dir)
        loaded = read_signals_from_dirs(self.test_dir, n_jobs=1)
        accepted = loaded["accepted"]
        for sig in accepted:
            self.assertEqual(sig.channel_names, ("C0", "C1", "C2"))

    def test_no_channel_names_file_uses_defaults(self):
        """When channel_names.txt is absent, signals get default names."""
        signals = self._make_signals(["X", "Y", "Z"])
        save_signals_to_dirs(signals, self.test_dir)
        # Remove the channel_names.txt
        os.remove(os.path.join(self.test_dir, "channel_names.txt"))
        loaded = read_signals_from_dirs(self.test_dir, n_jobs=1)
        accepted = loaded["accepted"]
        for sig in accepted:
            self.assertEqual(sig.channel_names, ("C0", "C1", "C2"))

    def test_roundtrip_signal_values(self):
        names = ["A", "B"]
        signals = RawSignals(sample_rate=500)
        data = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
        signals.append(RawSignal(signal=data, object_class="cls",
                                 channel_names=names, timestamp=0))
        save_signals_to_dirs(signals, self.test_dir)
        loaded = read_signals_from_dirs(self.test_dir, n_jobs=1)
        accepted = loaded["accepted"]
        self.assertEqual(len(accepted), 1)
        np.testing.assert_array_almost_equal(accepted[0].signal, data, decimal=4)
        self.assertEqual(accepted[0].channel_names, ("A", "B"))

    def test_empty_signals_no_channel_names_file(self):
        signals = RawSignals(sample_rate=1000)
        save_signals_to_dirs(signals, self.test_dir)
        cn_path = os.path.join(self.test_dir, "channel_names.txt")
        self.assertFalse(os.path.exists(cn_path))

    # --- Archive tests ---

    def _archive_path(self, name="test_data.zip"):
        return os.path.join(self.test_dir, name)

    def test_archive_channel_names_file_present(self):
        import zipfile
        signals = self._make_signals(["EMG1", "EMG2", "EMG3"])
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        with zipfile.ZipFile(path, "r") as z:
            cn_files = [n for n in z.namelist() if n.endswith("channel_names.txt")]
            self.assertEqual(len(cn_files), 1)

    def test_archive_channel_names_content(self):
        import zipfile
        names = ["EMG1", "EMG2", "EMG3"]
        signals = self._make_signals(names)
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        with zipfile.ZipFile(path, "r") as z:
            cn_files = [n for n in z.namelist() if n.endswith("channel_names.txt")]
            lines = [l.strip() for l in z.read(cn_files[0]).decode("utf-8").strip().splitlines()]
            self.assertEqual(lines, names)

    def test_archive_roundtrip_preserves_channel_names(self):
        names = ["aSensor_A", "aSensor_B", "aSensor_C"]
        signals = self._make_signals(names)
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        loaded = read_signals_from_archive(path)
        accepted = loaded["accepted"]
        self.assertTrue(len(accepted) > 0)
        for sig in accepted:
            self.assertEqual(sig.channel_names, tuple(names))

    def test_archive_roundtrip_default_channel_names(self):
        signals = self._make_signals(None, cols=3)
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        loaded = read_signals_from_archive(path)
        accepted = loaded["accepted"]
        for sig in accepted:
            self.assertEqual(sig.channel_names, ("C0", "C1", "C2"))

    def test_archive_roundtrip_signal_values(self):
        names = ["A", "B"]
        signals = RawSignals(sample_rate=500)
        data = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
        signals.append(RawSignal(signal=data, object_class="cls",
                                 channel_names=names, timestamp=0))
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        loaded = read_signals_from_archive(path)
        accepted = loaded["accepted"]
        self.assertEqual(len(accepted), 1)
        np.testing.assert_array_almost_equal(accepted[0].signal, data, decimal=4)
        self.assertEqual(accepted[0].channel_names, ("A", "B"))

    def test_archive_empty_signals_no_channel_names(self):
        import zipfile
        signals = RawSignals(sample_rate=1000)
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        with zipfile.ZipFile(path, "r") as z:
            cn_files = [n for n in z.namelist() if n.endswith("channel_names.txt")]
            self.assertEqual(len(cn_files), 0)

    def test_archive_many_channels(self):
        names = ["ch_{}".format(i) for i in range(50)]
        signals = self._make_signals(names, n=2, rows=5, cols=50)
        path = self._archive_path()
        save_signals_to_archive(signals, path)
        loaded = read_signals_from_archive(path)
        accepted = loaded["accepted"]
        for sig in accepted:
            self.assertEqual(sig.channel_names, tuple(names))


if __name__ == "__main__":
    unittest.main()
