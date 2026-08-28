"""Module defining the abstract base class for outlier generators.

Provides :class:`OutlierGenerator`, which establishes the interface for
generating synthetic outlier samples with configurable outlier labels.
"""
import abc
import numpy as np
from sklearn.utils.validation import check_is_fitted

class OutlierGenerator(abc.ABC):
    """Abstract base class for synthetic outlier generators.

    Subclasses must implement :meth:`fit` and :meth:`generate` to learn
    data characteristics and produce outlier samples respectively.

    Args:
        outlier_label: Label value assigned to generated outlier samples.

    """

    def __init__(self, outlier_label=-1) -> None:
        super().__init__()
        self.outlier_label_prototype = outlier_label

    @abc.abstractmethod
    def fit(self,X,y):
        """Fits Outlier generator

        Returns:
        self

        """
        np_labels = np.asanyarray(y)
        self.outlier_label_dtype_ = np_labels.dtype
        self.outlier_label_ = np.asanyarray([self.outlier_label_prototype]).astype(self.outlier_label_dtype_)
        

        return self



    @abc.abstractmethod
    def generate(self):
        """Generates outliers
        Returns:
        tuple X,y 
        """
        check_is_fitted(self, ("outlier_label_","outlier_label_dtype_"))
        return None

    def fit_generate(self,X,y):
        """Fit the generator and produce outliers in a single step.

        Args:
            X: Training feature matrix.
            y: Training label vector.

        Returns:
            Tuple of (outlier features, outlier labels).

        """
        return self.fit(X,y).generate()