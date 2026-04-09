import unittest

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.wt_aug.wt_aug_dwt import (
    WTAugDWT,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import (
    RawSignalsCreatorSines,
)


class WTAugDWTChecker(WTAugDWT, unittest.TestCase):

    def _apply_transformation(self, trans, decomp) -> list:
        transformed_coefficients = super()._apply_transformation(trans, decomp)
        self.assertIsNotNone(
            transformed_coefficients, "Transformed coefficients are None"
        )
        self.assertIsInstance(transformed_coefficients, list, "Transformed coefficients are not list")
        self.assertTrue(len(transformed_coefficients) == len(decomp), "Incompatible length of transformed coeffs")

        for coeffs, orig_coeffs in zip( transformed_coefficients, decomp):
            self.assertIsInstance(coeffs, np.ndarray, "Wrong type of coeffs")
            self.assertFalse(np.any(np.isnan(coeffs)), "Some coeffs are nans")
            self.assertFalse(np.any(np.isinf(coeffs)), "Some coeffs are infs")
            
            self.assertTupleEqual(coeffs.shape, orig_coeffs.shape, "Incompatible dimensions after transformation")
            self.assertTrue(coeffs.dtype == orig_coeffs.dtype, "Incompatible dtype asfer transformation")

        return transformed_coefficients


class IDecompTransformationTest(unittest.TestCase):

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise unittest.SkipTest("Skipping")

    def get_transformators(self) -> dict:
        raise unittest.SkipTest("Skipping")
    

    def generate_data(
        self, set_size=30, column_number=3, samples_number=100, dtype=np.float32
    ):
        signal_creator = RawSignalsCreatorSines(
            samples_number=samples_number,
            set_size=set_size,
            column_number=column_number,
            dtype=dtype,
        )
        raw_signals: RawSignals = signal_creator.get_set()
        return raw_signals

    def _check_raw_signals(self, raw_signals:RawSignals):

        for raw_signal in raw_signals:
            np_array = raw_signal.to_numpy()
            self.assertFalse(np.any(np.isinf(np_array)), "Infs in final transformation")
            self.assertFalse(np.any(np.isnan(np_array)), "NaNs in final transformation")

    def _compare_signals(self, orig_raw:RawSignals, trans_raw:RawSignals):
        for os, ts in zip(orig_raw,trans_raw):
            self.assertFalse(np.allclose(os.to_numpy(),ts.to_numpy()), "Transformed is too close to original")

    def test_transformation(self):
        transformers = self.get_transformators()

        configs = [(30, 3, 100), (1,3,100), (30, 1,100),]
        dtypes = [np.float32, np.float64]
        for trans_name, trans in transformers.items():
            for N, C, R in configs:
                for dtype in dtypes:
                    with self.subTest(
                        trans_name=trans_name, N=N, C=C, R=R, dtype=dtype
                    ):
                        raw_signals = self.generate_data(
                            set_size=N, column_number=C, samples_number=R, dtype=dtype
                        )
                        augmenter = WTAugDWTChecker(transformations=[trans])
                        t_raw_signals = augmenter.fit_transform(raw_signals)
                        self._check_raw_signals(t_raw_signals)
                        self._compare_signals(raw_signals, t_raw_signals)
