import random
from joblib import delayed
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import RawSignalsAugumenter
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from copy import deepcopy
import numpy as np

try:
    # Attempt to use the official library first
    from audiomentations import ClippingDistortion

except ImportError:
    # Fallback to the custom NumPy implementation if the library is missing
    print("audiomentations not found. Using custom ClippingDistortion fallback.")
    
    class ClippingDistortion:
        """
        A standalone, drop-in replacement for audiomentations.ClippingDistortion.
        """
        def __init__(
            self, 
            min_percentile_threshold=10.0, 
            max_percentile_threshold=30.0, 
            p=0.5
        ):
            self.min_percentile_threshold = min_percentile_threshold
            self.max_percentile_threshold = max_percentile_threshold
            self.p = p

        def __call__(self, samples, sample_rate):
            """
            Applies the clipping distortion.
            """
            # 1. Probability check
            if random.random() > self.p:
                return samples
                
            # 2. Pick a random percentile for this execution
            percentile = random.uniform(self.min_percentile_threshold, self.max_percentile_threshold)
            
            # 3. Calculate the amplitude bounds
            lower_bound = np.percentile(samples, percentile)
            upper_bound = np.percentile(samples, 100 - percentile)
            
            # 4. Clip the audio
            clipped_samples = np.clip(samples, lower_bound, upper_bound)
            
            return clipped_samples

from dexterous_bioprosthesis_2021_raw_datasets.tools.progressparallel import ProgressParallel

class RawSignalsAugumenterClippingDistortion(RawSignalsAugumenter):

    def __init__(self, min_percentile_threshold=10, max_percentile_threshold=50, n_repeats:int = 2, append_original=True, n_jobs=None) -> None:
        super().__init__()

        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.append_original = append_original

    def _sig_augument(self, signal:RawSignal, sample_rate):
        sig_list = []

        for _ in range(self.n_repeats):
                new_signal = deepcopy(signal)
                np_sig = new_signal.signal

                transformer = ClippingDistortion(p=0.8, min_percentile_threshold=self.min_percentile_threshold, max_percentile_threshold=self.max_percentile_threshold)
                
                for ch_id in range(np_sig.shape[1]):
                
                    ch_sig = np_sig[:,ch_id]
                    np_sig[:,ch_id] = transformer(ch_sig, sample_rate=sample_rate)

                sig_list.append(new_signal)

        return sig_list

    def fit(self, raw_signals: RawSignals):
        """
        Intentionally does nothing
        """
        return self

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        new_signals = raw_signals.initialize_empty()

        sample_rate = raw_signals.sample_rate
        

        aug_sig_list = ProgressParallel(n_jobs=self.n_jobs,use_tqdm=True,total=len(raw_signals))(delayed(self._sig_augument)( sig , sample_rate ) for sig in raw_signals  )

        for aug_sigs in aug_sig_list:
            new_signals+=aug_sigs    

        if self.append_original:
            new_signals+= raw_signals

        return new_signals

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        self.fit(raw_signals)
        return self.transform(raw_signals)
        