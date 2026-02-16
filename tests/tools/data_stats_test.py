import unittest

import numpy as np
import json

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.tools import data_stats


class DataStatsTest(unittest.TestCase):

    def generate_sample_data(
        self,
        signal_number=10,
        column_number=3,
        samples_number=12,
        dtype=np.float32,
        nclasses=3,
    ) -> RawSignals:
        signals = RawSignals()

        for i in range(1, signal_number + 1):
            obj_class = np.random.randint(nclasses)
            signals.append(
                RawSignal(
                    signal=np.random.random((samples_number, column_number)).astype(
                        dtype
                    ),
                    object_class=obj_class,
                )
            )

        return signals

    def test_data(self):

        signals = self.generate_sample_data()

        stats = data_stats.data_stats(signals)
        self.assertIsNotNone(stats, "Stats are none")
        self.assertIsInstance(stats, dict, "Wrong type")
        self.assertTrue(len(stats) > 1, "Length is too small.")

        json_str = json.dumps(stats)
        self.assertTrue(len(json_str)>1, "JSON too short")


if __name__ == "__main__":
    unittest.main()
