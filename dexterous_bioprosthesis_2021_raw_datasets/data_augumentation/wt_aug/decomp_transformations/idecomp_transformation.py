import abc

class IDecompTransformation(abc.ABC):

    @abc.abstractmethod
    def transform(self, decompositions:list)->list:
        """
        Transforms decomposition level

        """