import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator import (
    RawSignalsCreator,
)

import random


class RawSignalsCreatorSines(RawSignalsCreator):
    def __init__(
        self,
        set_size: int = 30,
        column_number: int = 3,
        samples_number: int = 100,
        class_indices=[0, 1, 2],
        noise_factor=0.1,
        phase_shift=np.pi,
        dtype=np.double,
    ) -> None:
        super().__init__()
        """
        Arguments:
        ---------
        set_size: int -- number of bjects in the dataset
        column_number:int -- number of columns (channels) in the signel
        samples_number: int -- number of samples per channel signal
        class_indices -- list of class indices 
        """

        self.set_size = set_size
        self.column_number = column_number
        self.samples_number = samples_number
        self.class_indices = class_indices
        self.noise_factor = noise_factor
        self.phase_shift = phase_shift
        self.dtype = dtype

    def get_set(self) -> RawSignals:
        signals = RawSignals()

        for i in range(1, self.set_size + 1):
            rnd_idx = random.randint(0, len(self.class_indices) - 1)
            rnd_class = self.class_indices[rnd_idx]
            noise = (
                np.random.rand(self.samples_number, self.column_number).astype(self.dtype)
                * self.noise_factor
            )
            sines_range = (rnd_idx + 1) * np.linspace(
                np.repeat(-2 * np.pi, self.column_number),
                np.repeat(2 * np.pi, self.column_number),
                num=self.samples_number,
                axis=1,
                dtype=self.dtype,
            )
            for n_col in range(self.column_number):
                p_shift = np.random.uniform(-self.phase_shift, self.phase_shift, size=1).astype(
                    self.dtype
                )
                sines_range[n_col,] += p_shift
            sine_signal = np.transpose(np.sin(sines_range))
            signal = noise + sine_signal

            signals.append(RawSignal(signal=signal, object_class=rnd_class))

        return signals
