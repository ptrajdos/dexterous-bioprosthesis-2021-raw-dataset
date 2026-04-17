import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class SamplerMixin:
    """
    Class enabling sampling for classes that implement fit and transform methods.
    """

    def sample(self, raw_signals: RawSignals, n_samples: int = 1) -> RawSignals:
        """
        Samples n_samples from the dataset
        Arguments:
        ---------
        raw_signals: RawSignals -- the dataset to be sampled
        n_samples: int -- how many samples to sample
        """
        n_pre_sampled_signals = len(raw_signals)
        rnd = self._random_state if hasattr(self, "_random_state") else np.random
        replace = n_samples > n_pre_sampled_signals
        indices = rnd.choice(n_pre_sampled_signals, size=n_samples, replace=replace)

        

        pre_sampled_signals = raw_signals.initialize_empty()
        for sig in raw_signals[indices]:
            pre_sampled_signals.append(sig)
        
        new_signals:RawSignals = self.transform(pre_sampled_signals)
        
        return new_signals
