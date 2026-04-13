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
        pre_sampled_signals = self.transform(raw_signals)  # type: ignore
        n_pre_sampled_signals = len(pre_sampled_signals)
        replace = n_samples > n_pre_sampled_signals
        rnd = self._random_state if hasattr(self, "_random_state") else np.random
        indices = rnd.choice(n_pre_sampled_signals, size=n_samples, replace=replace)
        new_signals = pre_sampled_signals.initialize_empty()
        for idx in indices:
            new_signals += [pre_sampled_signals[idx]]
        return new_signals
