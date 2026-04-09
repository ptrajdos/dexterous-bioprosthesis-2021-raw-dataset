from __future__ import annotations
from typing import Optional
import numpy as np
import abc
from sklearn.exceptions import NotFittedError
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.decomp_transformations.decomp_transformation_dummy import (
    DecompTransformationDummy,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class WTAugBase(RawSignalsAugumenter):

    def __init__(
        self,
        wavelets: Optional[list] = None,
        max_decomposition_level: int = 3,
        transformations: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.wavelets = wavelets
        self.max_decomposition_level = max_decomposition_level
        self.transformations = transformations

    def _set_effective_wavelets(self) -> None:
        if self.wavelets is None:
            self._effective_wavelets = ["db4", "db6"]
        else:
            self._effective_wavelets = self.wavelets

    def _set_effective_transformations(self) -> None:
        if self.transformations is None:
            self._effective_transformations: list = [DecompTransformationDummy()]
        else:
            self._effective_transformations: list = self.transformations

    def fit(self, raw_signals: RawSignals):
        self._set_effective_wavelets()
        self._set_effective_transformations()
        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def _select_params(self, raw_signals: RawSignals) -> tuple:
        n_signals = len(raw_signals)
        sel_wavelets = np.random.choice(
            self._effective_wavelets, size=n_signals, replace=True
        )
        sel_levels = np.random.choice(
            np.arange(1,self.max_decomposition_level+1), size=n_signals, replace=True
        )
        sel_transformations = np.random.choice(
            self._effective_transformations, size=n_signals, replace=True
        )

        return (sel_wavelets, sel_levels, sel_transformations)
    
    @abc.abstractmethod
    def _wt_trans(self, raw_signal:RawSignal, wavelet, level:int)->list:
        """
        Transforms a signal using a kind of wavelet transform
        """

    @abc.abstractmethod
    def _wt_itrans(self, raw_signal:RawSignal, wavelet, decomps:list)->RawSignal:
        """
        Inverse transformation
        """
        
    def _apply_transformation(self,trans,decomp)->list:
        return trans.transform(decomp)

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        self._check_fitted()
        sel_wavelets, sel_levels, sel_transformations = self._select_params(raw_signals)
        new_signals = raw_signals.initialize_empty()

        for wav, lvl, trans, sig in zip(sel_wavelets, sel_levels, sel_transformations, raw_signals):
            decomp = self._wt_trans(sig, wav, lvl)
            transformed = self._apply_transformation(trans, decomp)
            t_sig = self._wt_itrans(sig,wav,transformed)
            new_signals.append(t_sig)

        return new_signals

    def sample(self, raw_signals: RawSignals, n_samples: int = 1) -> RawSignals:
        self._check_fitted()
        n_sigs = len(raw_signals)
        replace = n_samples > n_sigs
        indices = np.random.choice(n_sigs, n_samples, replace=replace)
        sampled_signals = raw_signals.initialize_empty()
        sampled_signals += raw_signals[indices]
        return self.transform(sampled_signals)

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        self.fit(raw_signals)
        return self.transform(raw_signals)
