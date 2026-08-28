"""Module defining the abstract interface for signal relabelers.

Provides :class:`Relabeler` with fit/transform pattern.
"""
import abc

class Relabeler(abc.ABC):
    """Abstract interface for signal relabelers."""

    @abc.abstractmethod
    def fit(self,labels):
        """Fits the object

        Arguments:
        ---------
        labels -- label objects to fit with.

        Returns:
        -------
          self

        """
    @abc.abstractmethod
    def transform(self, labels):
        """Transforms the labels

        Arguments:
        ---------
        labels -- label objects to transform

        Returns:
        new labels

        """

    def fit_transform(self, labels):
        """Fit and then transform the object with given labels

        Arguments:
        ---------
        labels -- labels to fit object with and then transforms

        Returns:
        -------
        new labels

        """
        return self.fit(labels).transform(labels)