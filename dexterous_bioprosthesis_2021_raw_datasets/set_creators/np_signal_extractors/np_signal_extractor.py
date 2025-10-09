import abc

import numpy as np
import logging

class NPSignalExtractor(abc.ABC):

    def _check_input(self, X):
        """
        Check the input data.
        """
        assert not np.any(
            np.isnan(X)
        ), "Input data contains NaN values. Please clean the data before processing."
        assert not np.isinf(
            X
        ).any(), "Input data contains infinite values. Please clean the data before processing."
        return True

    def _sanitize_output(self, X):
        if self.sanitize_output:
            return np.nan_to_num(X, nan=0.0)

        return X

    def _check_output(self, X):
        """
        Check the output data.
        """
        assert not np.any(
            np.isnan(X)
        ), "Output data contains NaN values. Please clean the data before processing."
        assert not np.isinf(
            X
        ).any(), "Output data contains infinite values. Please clean the data before processing."
        return True

    def __init__(self, sanitize_output=False, check_input=False, check_output=False):
        """
        Initialize the signal extractor.
        Parameters
        ----------
        sanitize_output : bool, optional
            If True, sanitize the output data.
        check_input : bool, optional
            If True, check the input data for NaN and infinite values.
        check_output : bool, optional
            If True, check the output data for NaN and infinite values.
        """
        self.sanitize_output = sanitize_output
        self.check_input = check_input
        self.check_output = check_output

    def fit(self, X, fs=1000):
        if self.check_input:
            self._check_input(X)

        self._fs = fs

        return self

    @abc.abstractmethod
    def _transform(self, X):
        """
        Transform the input data.
        """

    def transform(self, X):
        if self.check_input:
            self._check_input(X)
        X_t = self._transform(X)

        if not hasattr(self, "_fs"):
            logging.warning("Transformeer is not fitted. In future releases an exception will be raised.")
            # raise ValueError("Transformeer is not fitted")
        
        if self.check_output:
            self._check_output(X_t)

        if self.sanitize_output:
            return self._sanitize_output(X_t)

        return X_t

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @abc.abstractmethod
    def attribs_per_column(self):
        pass
