"""Module implementing clipping distortion augmentation for raw signals.

Applies amplitude clipping to each channel of a signal, simulating
sensor saturation or distortion artifacts. Uses the ``audiomentations``
library when available, otherwise falls back to a custom NumPy implementation.
"""
import random
from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signal_augumenter_base import (
    RawSignalsAugumenterBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal

try:
    # Attempt to use the official library first
    from audiomentations import ClippingDistortion

except ImportError:
    # Fallback to the custom NumPy implementation if the library is missing
    print("audiomentations not found. Using custom ClippingDistortion fallback.")

    class ClippingDistortion:
        """A standalone, drop-in replacement for audiomentations.ClippingDistortion.
        """

        def __init__(
            self, min_percentile_threshold=10.0, max_percentile_threshold=30.0, p=0.5
        ):
            self.min_percentile_threshold = min_percentile_threshold
            self.max_percentile_threshold = max_percentile_threshold
            self.p = p

        def __call__(self, samples, sample_rate):
            """Applies the clipping distortion.
            """
            # 1. Probability check
            if random.random() > self.p:
                return samples

            # 2. Pick a random percentile for this execution
            percentile = random.uniform(
                self.min_percentile_threshold, self.max_percentile_threshold
            )

            # 3. Calculate the amplitude bounds
            lower_bound = np.percentile(samples, percentile)
            upper_bound = np.percentile(samples, 100 - percentile)

            # 4. Clip the audio
            clipped_samples = np.clip(samples, lower_bound, upper_bound)

            return clipped_samples




class RawSignalsAugumenterClippingDistortion(RawSignalsAugumenterBase):
    """Augmenter that applies clipping distortion to raw signals.

    Clips signal amplitudes at percentile-based thresholds independently
    for each channel, producing distorted versions of the original signals.

    Args:
        min_percentile_threshold: Lower bound for the random percentile threshold.
        max_percentile_threshold: Upper bound for the random percentile threshold.
        n_repeats: Number of augmented copies per signal.
        append_original: Whether to include original signals in the output.
        n_jobs: Number of parallel jobs.
        random_state: Random seed for reproducibility.

    """

    def __init__(
        self,
        min_percentile_threshold=10,
        max_percentile_threshold=50,
        n_repeats: int = 2,
        append_original=True,
        n_jobs=None,
        random_state=10,
    ) -> None:
        super().__init__(
            n_jobs=n_jobs,
            append_original=append_original,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold

    def _sig_augument(self, raw_signal: RawSignal, n_repeats: int = 1):
        """Apply clipping distortion to a single signal.

        Args:
            raw_signal: The signal to augment.
            n_repeats: Number of augmented versions to create.

        Returns:
            List of clipping-distorted signals.

        """
        sample_rate = raw_signal.sample_rate
        sig_list = []

        for _ in range(n_repeats):
            new_signal = deepcopy(raw_signal)
            np_sig = new_signal.signal

            transformer = ClippingDistortion(
                p=0.8,
                min_percentile_threshold=self.min_percentile_threshold,
                max_percentile_threshold=self.max_percentile_threshold,
            )

            for ch_id in range(np_sig.shape[1]):

                ch_sig = np_sig[:, ch_id]
                np_sig[:, ch_id] = transformer(ch_sig, sample_rate=sample_rate)

            sig_list.append(new_signal)

        return sig_list
