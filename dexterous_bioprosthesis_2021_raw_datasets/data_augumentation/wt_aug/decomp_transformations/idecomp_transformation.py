from __future__ import annotations
import abc

class IDecompTransformation(abc.ABC):

    @abc.abstractmethod
    def fit(self)->IDecompTransformation:
        """
        Just for initialization
        """

    @abc.abstractmethod
    def transform(self, decompositions:list)->list:
        """
        Transforms decomposition level

        """

    def fit_transform(self, decompositions:list):
        return self.fit().transform(decompositions)