from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor import (
    NPSignalExtractor,
)

import numpy as np


class NpSignalExtractorTemporalMoment(NPSignalExtractor):

    def __init__(
        self,
        sanitize_output=False,
        check_input=False,
        check_output=False,
        order=1,
        central=True,
        proportional_time=True,
    ):
        super().__init__(sanitize_output, check_input, check_output)
        self.order = order
        self.central = central
        self.proportional_time = proportional_time

    @staticmethod
    def _get_time_vector(X, proportional_time=True):
        n_samples = X.shape[0]
        if proportional_time:
            return np.linspace(0, 1, n_samples, dtype=X.dtype)
        else:
            return np.arange(n_samples, dtype=X.dtype)

    @staticmethod
    def _get_X_normalized_columns(X):
        col_sums = np.sum(X, axis=0)
        mask = np.isclose(col_sums, 0)
        col_sums[mask] = 1.0
        return X / col_sums

    @staticmethod
    def _calculate_moment(X, order, time_vector, central):

        if order < 0:
            raise ValueError("Order must be non-negative")

        if order == 0:
            return np.ones(X.shape[1], dtype=X.dtype)

        if order == 1:
            result = time_vector @ X
            return result

        if central:
            first_moment = NpSignalExtractorTemporalMoment._calculate_moment(
                X, 1, time_vector, False
            )
            time_vectors = time_vector[:, np.newaxis] - first_moment
            time_vector_powered = time_vectors**order
            result = np.sum(time_vector_powered * X, axis=0)
        else:
            time_vectors = time_vector
            time_vector_powered = time_vectors**order
            result = time_vector_powered @ X

        return result

    def _transform(self, X):

        return self._calculate_moment(
            X=NpSignalExtractorTemporalMoment._get_X_normalized_columns(X),
            order=self.order,
            time_vector=NpSignalExtractorTemporalMoment._get_time_vector(
                X, self.proportional_time
            ),
            central=self.central,
        )

    def attribs_per_column(self):
        return 1
