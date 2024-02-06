import numpy as np
import random
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


def generate_sample_data(signal_number=10, column_number=3, samples_number=12, class_indices=[0,1])->RawSignals:
        signals = RawSignals()

        for i in range(1,signal_number+1):
            rnd_idx = random.randint(0, len(class_indices)-1)
            rnd_class = class_indices[rnd_idx]
            signals.append( RawSignal(signal=np.random.rand( samples_number,column_number), object_class=rnd_class ) )

        return signals