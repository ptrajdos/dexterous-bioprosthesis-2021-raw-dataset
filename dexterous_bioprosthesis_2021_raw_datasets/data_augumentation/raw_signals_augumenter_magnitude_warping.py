"""Module implementing magnitude warping augmentation for raw signals.

Applies a smooth, randomly generated warping curve to the signal amplitude.
The curve is constructed by fitting a cubic smoothing spline through
randomly placed knots, producing non-uniform scaling along the time axis.
"""
from copy import deepcopy

import numpy as np
from csaps import csaps

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signal_augumenter_base import (
    RawSignalsAugumenterBase,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal


class RawSignalsAugumenterMagnitudeWarping(RawSignalsAugumenterBase):
    """Augmenter that applies smooth magnitude warping to raw signals.

    Generates a random smooth curve via cubic spline interpolation and
    multiplies it element-wise with the signal, creating non-uniform
    amplitude scaling over time.

    Args:
        n_repeats: Number of augmented copies per signal.
        append_original: Whether to include original signals in the output.
        n_jobs: Number of parallel jobs.
        random_state: Random seed for reproducibility.
        min_knots: Minimum number of spline knots.
        max_knots: Maximum number of spline knots.
        scale: Scaling factor for the warping curve standard deviation,
            relative to the global signal standard deviation.
        smooth: Smoothing parameter for the cubic spline (0 to 1).

    """

    def __init__(
        self,
        n_repeats: int = 2,
        append_original=True,
        n_jobs=None,
        random_state=10,
        min_knots:int = 2,
        max_knots:int = 5,
        scale:float = 0.05,
        smooth:float = 0.8,
    ) -> None:
        super().__init__(
            n_jobs=n_jobs,
            append_original=append_original,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        self.min_knots = min_knots
        self.max_knots = max_knots
        self.scale = scale
        self.smooth = smooth

    def _sig_augument(self, raw_signal: RawSignal, n_repeats: int = 1):
        """Apply magnitude warping to a single signal.

        Args:
            raw_signal: The signal to augment.
            n_repeats: Number of augmented versions to create.

        Returns:
            List of magnitude-warped signals.

        """
        sig_list = []
        fs = raw_signal.get_sample_rate()
        base_sig_np = raw_signal.to_numpy()
        n_rows, n_cols = base_sig_np.shape

        global_sd = np.std(raw_signal.to_numpy())
        e_sd = global_sd * self.scale
        t = np.arange(n_rows)/fs
        
        for _ in range(n_repeats):
            new_signal = deepcopy(raw_signal)
            np_sig = new_signal.signal
            
            e_knots_n = self._random_state.randint(self.min_knots, self.max_knots+1)
            knot_t = np.linspace(t[0], t[-1], e_knots_n)
            knot_y = self._random_state.normal(loc=1.0, scale=e_sd, size=e_knots_n)
            wrap_curve = csaps(knot_t, knot_y, t, smooth=self.smooth)

            np_sig*= wrap_curve[:,None] # type: ignore
            

            sig_list.append(new_signal)

        return sig_list
