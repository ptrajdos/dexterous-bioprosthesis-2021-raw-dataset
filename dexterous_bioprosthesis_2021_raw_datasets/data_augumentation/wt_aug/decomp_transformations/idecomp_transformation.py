"""Module defining the abstract interface for decomposition transformations.

Provides :class:`IDecompTransformation`, the contract that all wavelet
coefficient transformations must implement.
"""
from __future__ import annotations
import abc

class IDecompTransformation(abc.ABC):
    """Abstract interface for wavelet decomposition coefficient transformations.

    Implementations must provide :meth:`fit` and :meth:`transform` methods
    to initialise and apply transformations to decomposition coefficient lists.
    """

    @abc.abstractmethod
    def fit(self)->IDecompTransformation:
        """Just for initialization
        """

    @abc.abstractmethod
    def transform(self, decompositions:list)->list:
        """Transforms decomposition level

        """

    def fit_transform(self, decompositions:list):
        """Fit and then transform the decomposition coefficients.

        Args:
            decompositions: List of wavelet decomposition coefficient arrays.

        Returns:
            Transformed list of coefficient arrays.

        """
        return self.fit().transform(decompositions)