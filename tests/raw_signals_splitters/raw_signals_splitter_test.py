import unittest
from typing import Dict
import abc

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_splitter.iraw_signals_splitter import IRawSignalsSplitter


class RawSignalsSplitterTest(unittest.TestCase):
    __test__ = False

    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise unittest.SkipTest("Skipping")

    @abc.abstractmethod
    def get_splitters(self) -> Dict[str, IRawSignalsSplitter]:
        raise unittest.SkipTest("Skipping")

    def generate_sample_data(
            self,
            signal_number=10,
            column_number=3,
            samples_number=12,
            dtype=np.float32,
            labels=[0, 1, 2],
    ) -> RawSignals:
        signals = RawSignals()

        for i in range(1, signal_number + 1):
            label = np.random.choice(labels, 1)
            signals.append(
                RawSignal(
                    signal=np.random.random((samples_number, column_number)).astype(
                        dtype
                    ),
                    object_class=label,
                )
            )

        return signals

    def test_splitting_simple(self):

        for spliter_name, spliter in self.get_splitters().items():
            test_data = self.generate_sample_data()
            with self.subTest(spliter_name=spliter_name):
                splitted = spliter.split_signals(test_data)

                self.assertIsNotNone(splitted, "splitted signals should not be None")
                for splitted_signal in splitted:
                    self.assertIsNotNone(splitted_signal, "splitted signal should not be None")
                    self.assertIsInstance(splitted_signal, RawSignals,
                                          "splitted signal should be an instance of RawSignals")
