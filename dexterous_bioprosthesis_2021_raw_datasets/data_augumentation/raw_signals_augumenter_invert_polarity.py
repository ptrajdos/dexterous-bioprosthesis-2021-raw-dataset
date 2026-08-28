"""Module implementing polarity inversion augmentation for raw signals.

Multiplies all signal values by -1, effectively flipping the signal
around the zero axis. This is a simple but effective augmentation for
symmetric signal distributions.
"""
from copy import deepcopy


from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signal_augumenter_base import (
    RawSignalsAugumenterBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal


class RawSignalsAugumenterInvertPolarity(RawSignalsAugumenterBase):
    """Augmenter that inverts the polarity of raw signals.

    Produces a single inverted copy of each signal by multiplying all
    sample values by -1.

    Args:
        append_original: Whether to include original signals in the output.
        n_jobs: Number of parallel jobs.
        n_repeats: Number of augmented copies per signal (polarity inversion
            always produces one copy regardless of this value).
        random_state: Random seed for reproducibility.

    """

    def __init__(
        self, append_original=True, n_jobs=None, n_repeats: int = 1, random_state=10
    ) -> None:
        super().__init__(
            n_jobs=n_jobs,
            append_original=append_original,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    def _sig_augument(self, raw_signal: RawSignal, n_repeats: int = 1):
        """Invert the polarity of a single signal.

        Args:
            raw_signal: The signal to augment.
            n_repeats: Ignored; always produces one inverted copy.

        Returns:
            List containing a single polarity-inverted signal.

        """
        new_signal = deepcopy(raw_signal)
        np_sig = new_signal.signal

        np_sig *= -1.0

        return [new_signal]
