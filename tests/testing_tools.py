import os
import warnings
import numpy as np
import random
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


def generate_sample_data(signal_number=10, column_number=3, samples_number=12, class_indices=[0,1], dtype=np.double)->RawSignals:
        signals = RawSignals()

        for i in range(1,signal_number+1):
            rnd_idx = random.randint(0, len(class_indices)-1)
            rnd_class = class_indices[rnd_idx]
            signals.append( RawSignal(signal=np.random.rand( samples_number,column_number).astype(dtype), object_class=rnd_class ) )

        return signals

def generate_zero_data(signal_number=10, column_number=3, samples_number=12, class_indices=[0,1], dtype=np.double)->RawSignals:
        signals = RawSignals()

        for i in range(1,signal_number+1):
            rnd_idx = random.randint(0, len(class_indices)-1)
            rnd_class = class_indices[rnd_idx]
            signals.append( RawSignal(signal=np.zeros( (samples_number,column_number),dtype=dtype), object_class=rnd_class ) )

        return signals

import pickle
from tempfile import mkstemp
import unittest

def get_pickled_obj(obj):
        pickle_file_path = mkstemp()[1]
        piclke_file_handler = open(pickle_file_path,'wb')
        pickle.dump(obj, piclke_file_handler)
        piclke_file_handler.close()

        file_stats = os.stat(pickle_file_path)
        file_size = file_stats.st_size
        assert file_size > 0

        piclke_file_handler  = open(pickle_file_path,'rb')
        obj_pickle = pickle.load(piclke_file_handler)
        piclke_file_handler.close()
        try:
            os.remove(pickle_file_path)
        except Exception as ex:
            warnings.warn("Cannot remove the pickle file: {}".format(ex))

        return obj_pickle