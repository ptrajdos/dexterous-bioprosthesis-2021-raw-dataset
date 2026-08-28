"""Module implementing additive white noise augmentation for raw signals.

Adds Gaussian white noise scaled by a random fraction of each channel's
standard deviation, simulating sensor noise or environmental interference.
"""
from copy import deepcopy

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signal_augumenter_base import (
    RawSignalsAugumenterBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class RawSignalsAugumenterWhiteNoise(RawSignalsAugumenterBase):
    """Augmenter that adds white Gaussian noise to raw signals.

    Noise amplitude is scaled per channel based on the channel's standard
    deviation and a random percentage drawn from ``[noise_perc_min, noise_perc_max]``.

    Args:
        noise_perc_min: Minimum noise percentage relative to channel std.
        noise_perc_max: Maximum noise percentage relative to channel std.
        n_repeats: Number of augmented copies per signal.
        append_original: Whether to include original signals in the output.
        n_jobs: Number of parallel jobs.
        random_state: Random seed for reproducibility.

    """

    def __init__(
        self,
        noise_perc_min=0.01,
        noise_perc_max=1.0,
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

        self.noise_perc_min = noise_perc_min
        self.noise_perc_max = noise_perc_max

    def _sig_augument(self, raw_signal: RawSignal, n_repeats: int = 1) -> list:
        """Add white noise to a single signal.

        Args:
            raw_signal: The signal to augment.
            n_repeats: Number of augmented versions to create.

        Returns:
            List of noise-augmented signals.

        """
        sig_list = []

        orig_sig = raw_signal.signal
        n_samples, n_channels = orig_sig.shape

        for _ in range(n_repeats):
            new_signal = deepcopy(raw_signal)

            noise_perc = self._random_state.uniform(
                self.noise_perc_min, self.noise_perc_max, (1, n_channels)
            )
            stds = orig_sig.std(axis=0, keepdims=True)  # shape (1, n_channels)
            noise = self._random_state.normal(0, 1, (n_samples, n_channels)) * stds
            new_signal.signal += noise_perc * noise
            sig_list.append(new_signal)

        return sig_list
