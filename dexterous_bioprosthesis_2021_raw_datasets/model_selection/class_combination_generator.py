"""Module providing a generator for class subset combinations.

Contains :class:`ClassCombinationGenerator`, which yields index arrays
for every possible combination of a given number of classes, useful for
pairwise or multi-class evaluation strategies.
"""
import itertools
from scipy.special import binom
import numpy as np
from sklearn.exceptions import NotFittedError

class ClassCombinationGenerator:
    """Generator that yields sample indices for all class combinations.

    Given a label vector, produces every possible subset of
    ``num_to_select`` classes and yields the corresponding sample indices.
    """

    def __init__(self, num_to_select:int = 2) -> None:
        """Generates sets with given number of classes.
        Every possible combination of classes is generated.

        Arguments:
        ---------
        num_to_select: int -- the number of class to be selected in each subset

        """
        self.num_to_select = num_to_select
        self.y = None
        self.n_combinations = None

    def fit(self,X,y):
        """Fit the generator by storing labels and computing combination count.

        Args:
            X: Feature matrix (not used, present for API compatibility).
            y: Label vector.

        """
        self.y = y
        n_classes = len(np.unique(y))
        self.n_combinations  = binom(n_classes, self.num_to_select)

    def get_indices(self):
        """Yield index arrays for each class combination.

        Yields:
            numpy array of sample indices belonging to the current
            class combination.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.

        """
        if self.y is None:
            raise NotFittedError("Model is not fitted")

        u_classes = np.unique(self.y)
        n_objects = len(self.y)

        for class_group in itertools.combinations(u_classes, self.num_to_select):
            selection_mask = np.zeros( n_objects, dtype=np.bool_)
            for class_label in class_group:
                selection_mask = np.bitwise_or(selection_mask, self.y == class_label)

            indices = np.asanyarray(np.nonzero(selection_mask)[0])
            yield indices
