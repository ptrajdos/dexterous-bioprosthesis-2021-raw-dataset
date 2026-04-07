from sklearn.dummy import check_random_state

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)


class RawSignalsFilterObjResampler(RawSignalsFilter):

    def __init__(
        self,
        resampling_rate: float = 1.0,
        with_replacement: bool = False,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.resampling_rate = resampling_rate
        self.with_replacement = with_replacement
        self.random_state = random_state

    def fit(self, raw_signals: RawSignals):
        """
        Only rng
        """
        self.rng_ = check_random_state(self.random_state)
        return self

    def _calculate_n_resampled(self, raw_signals: RawSignals):
        n_samples = len(raw_signals)

        n_resampled = max(1, int(n_samples * self.resampling_rate))
        if n_resampled > n_samples and not self.with_replacement:
            raise ValueError(
                f"Cannot resample {n_resampled} samples from {n_samples} without replacement."
            )

        return n_resampled

    def transform(self, raw_signals: RawSignals):
        """
        Resample
        """
        if not hasattr(self, "rng_"):
            raise ValueError("The filter has not been fitted yet.")

        n_resampled = self._calculate_n_resampled(raw_signals)

        resampled_indices = self.rng_.choice(
            len(raw_signals), size=n_resampled, replace=self.with_replacement
        )
        resampled_signals = raw_signals[resampled_indices]
        return resampled_signals
