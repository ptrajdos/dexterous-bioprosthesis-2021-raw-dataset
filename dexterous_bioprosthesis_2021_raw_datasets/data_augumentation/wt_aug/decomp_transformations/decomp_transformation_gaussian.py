"""Module implementing additive Gaussian noise for decomposition coefficients.

Adds Gaussian noise scaled by each coefficient level's standard deviation
to the wavelet decomposition, optionally skipping approximation coefficients.
"""
import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_base import (
    DecompTransformationBase,
)


class DecompTransformationGaussian(DecompTransformationBase):
    """Transformation that adds Gaussian noise to decomposition coefficients.

    Noise amplitude is scaled per channel based on the coefficient standard
    deviation and a random percentage.

    Args:
        mean: Mean of the Gaussian noise.
        min_noise_perc: Minimum noise percentage relative to coefficient std.
        max_noise_perc: Maximum noise percentage relative to coefficient std.
        alter_approximation_coeffs: If ``True``, also modify the approximation
            (lowest frequency) coefficients.
        random_state: Random seed for reproducibility.

    """

    def __init__(
        self,
        mean=0,
        min_noise_perc=0.01,
        max_noise_perc=0.1,
        alter_approximation_coeffs=False,
        random_state=10,
    ) -> None:
        super().__init__(random_state)
        self.mean = mean
        self.min_noise_perc = min_noise_perc
        self.max_noise_perc = max_noise_perc
        self.alter_approximation_coeffs = alter_approximation_coeffs

    def transform(self, decompositions: list):
        """Add Gaussian noise to decomposition coefficients.

        Args:
            decompositions: List of wavelet coefficient arrays.

        Returns:
            List of noise-augmented coefficient arrays.

        """
        self._check_if_fitted()
        new_decomps = []

        for coeff_idx, coeff in enumerate(decompositions):
            new_coeffs = coeff.copy()
            if coeff_idx == 0 and not self.alter_approximation_coeffs:
                new_decomps.append(new_coeffs)
                continue

            n_samples, n_channels = coeff.shape
            noise_perc = self._random_state.uniform(
                self.min_noise_perc, self.max_noise_perc, (1, n_channels)
            )
            stds = coeff.std(axis=0, keepdims=True)  # shape (1, n_channels)
            noise = self._random_state.normal(0, 1, (n_samples, n_channels)) * stds
            new_coeffs += noise_perc * noise
            new_decomps.append(new_coeffs)

        return new_decomps
