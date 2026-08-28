"""Module implementing uniform multiplicative scaling for decomposition coefficients.

Scales each decomposition level by a random factor drawn from a uniform
distribution, optionally skipping approximation coefficients.
"""
import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_base import (
    DecompTransformationBase,
)


class DecompTransformationMultiplierUniform(DecompTransformationBase):
    """Transformation that scales decomposition coefficients by a uniform random factor.

    Each detail coefficient level is multiplied by a per-channel random
    factor drawn from ``[min_noise_perc, max_noise_perc]``.

    Args:
        min_noise_perc: Minimum multiplier value.
        max_noise_perc: Maximum multiplier value.
        alter_approximation_coeffs: If ``True``, also scale the approximation
            (lowest frequency) coefficients.
        random_state: Random seed for reproducibility.

    """

    def __init__(
        self,
        min_noise_perc=0.8,
        max_noise_perc=1.2,
        alter_approximation_coeffs=False,
        random_state=10,
    ) -> None:
        super().__init__(random_state=random_state)
        self.min_noise_perc = min_noise_perc
        self.max_noise_perc = max_noise_perc
        self.alter_approximation_coeffs = alter_approximation_coeffs

    def transform(self, decompositions: list):
        """Scale decomposition coefficients by uniform random factors.

        Args:
            decompositions: List of wavelet coefficient arrays.

        Returns:
            List of scaled coefficient arrays.

        """
        self._check_if_fitted()
        new_decomps = []

        for coeff_idx, coeff in enumerate(decompositions):
            new_coeffs = coeff.copy()
            if coeff_idx == 0 and not self.alter_approximation_coeffs:
                new_decomps.append(new_coeffs)
                continue

            n_samples, n_channels = coeff.shape
            noise_mult = self._random_state.uniform(
                self.min_noise_perc, self.max_noise_perc, (1, n_channels)
            )
            new_coeffs *= noise_mult
            new_decomps.append(new_coeffs)

        return new_decomps
