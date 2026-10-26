"""Module implementing energy-over-entropy ratio extraction.

Computes the ratio of signal energy to energy entropy.
"""
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_energy_entropy import (
    NpSignalExtractorEnergyEntropy,
)

import numpy as np


class NpSignalExtractorEnergyOverEntropy(NPSignalExtractor):
    """Extractor that computes the energy-over-entropy ratio of each channel.

    Energy is the time integral of the squared signal, ``sum(x**2) / fs``,
    where ``fs`` is the sample rate passed to ``fit``. Entropy is the energy
    entropy computed by :class:`NpSignalExtractorEnergyEntropy`.

    Channels with zero entropy (zero energy, or all energy in a single
    sample) yield 0.
    """

    def _transform(self, X):
        fs = self._get_fs()
        energy = np.sum(X ** 2, axis=0) / fs
        entropy = NpSignalExtractorEnergyEntropy()._transform(X)

        eoe = np.divide(
            energy, entropy, out=np.zeros_like(energy), where=entropy > 0
        )
        return eoe.astype(X.dtype)

    def attribs_per_column(self):
        """Return the number of features extracted per channel."""
        return 1
