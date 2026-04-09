
import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.idecomp_transformation import (
    IDecompTransformation,
)


class DecompTransformationMultiplierUniform(IDecompTransformation):

    def __init__(
        self,
        min_noise_perc=0.8,
        max_noise_perc=1.2,
        alter_approximation_coeffs=False,
    ) -> None:
        super().__init__()
        self.min_noise_perc = min_noise_perc
        self.max_noise_perc = max_noise_perc
        self.alter_approximation_coeffs = alter_approximation_coeffs

    def transform(self, decompositions: list):
        new_decomps = []

        for coeff_idx, coeff in enumerate(decompositions):
            new_coeffs = coeff.copy()
            if coeff_idx == 0 and not self.alter_approximation_coeffs:
                new_decomps.append(new_coeffs)
                continue

            n_samples, n_channels = coeff.shape
            noise_mult = np.random.uniform(
                self.min_noise_perc, self.max_noise_perc, (1, n_channels)
            )            
            new_coeffs *= noise_mult
            new_decomps.append(new_coeffs)

        return new_decomps
