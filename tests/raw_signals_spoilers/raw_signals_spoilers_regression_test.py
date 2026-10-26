"""Regression tests for spoilers: SNR accuracy, reproducibility, edge cases."""
import unittest
import warnings

import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_cubicclipper import (
    RawSignalsSpoilerCubicClipper,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_damper import (
    RawSignalsSpoilerDamper,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_dummy import (
    RawSignalsSpoilerDummy,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_gauss import (
    RawSignalsSpoilerGauss,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_multiple import (
    RawSignalsSpoilerMultiple,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_rappam import (
    RawSignalsSpoilerRappAM,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_sine import (
    RawSignalsSpoilerSine,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_sinesh import (
    RawSignalsSpoilerSinesHarmonics,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_tanhclipper import (
    RawSignalsSpoilerTanhClipper,
)

SNR_SPOILERS = [
    RawSignalsSpoilerGauss,
    RawSignalsSpoilerSine,
    RawSignalsSpoilerSinesHarmonics,
    RawSignalsSpoilerDamper,
    RawSignalsSpoilerCubicClipper,
    RawSignalsSpoilerTanhClipper,
    RawSignalsSpoilerRappAM,
]


def make_signals(scale=1.0, n=3, length=1000, n_channels=4, dtype=float):
    rng = np.random.default_rng(1)
    signals = RawSignals(sample_rate=1000)
    for i in range(n):
        data = (rng.normal(size=(length, n_channels)) * scale).astype(dtype)
        signals.append(RawSignal(data, object_class=i))
    return signals


def achieved_snrs(clean, spoiled):
    snrs = []
    for c, s in zip(clean, spoiled):
        x = c.to_numpy().astype(float)
        noise_power = np.mean((s.to_numpy() - x) ** 2, axis=0)
        mask = noise_power > 0
        snrs += list(10 * np.log10(np.mean(x**2, axis=0)[mask] / noise_power[mask]))
    return np.asarray(snrs)


class RawSignalsSpoilersRegressionTest(unittest.TestCase):

    def test_target_snr_any_amplitude(self):
        for cls in SNR_SPOILERS:
            for scale in [1e-4, 1.0, 1000.0]:
                with self.subTest(spoiler=cls.__name__, scale=scale):
                    clean = make_signals(scale)
                    spoiled = cls(snr=10, channels_spoiled_frac=0.25).fit_transform(clean)
                    snrs = achieved_snrs(clean, spoiled)
                    self.assertEqual(len(snrs), len(clean))
                    self.assertTrue(np.allclose(snrs, 10, atol=0.05), snrs)

    def test_reproducible(self):
        for cls in SNR_SPOILERS:
            with self.subTest(spoiler=cls.__name__):
                a = cls(random_state=0).fit_transform(make_signals())
                b = cls(random_state=0).fit_transform(make_signals())
                self.assertEqual(a, b)

    def test_integer_signals(self):
        for cls in SNR_SPOILERS:
            with self.subTest(spoiler=cls.__name__):
                spoiled = cls(channels_spoiled_frac=1.0).fit_transform(
                    make_signals(scale=100, dtype=np.int16)
                )
                self.assertTrue(np.issubdtype(spoiled[0].to_numpy().dtype, np.floating))

    def test_damper_negative_snr_does_not_invert(self):
        clean = make_signals()
        with self.assertWarns(UserWarning):
            spoiled = RawSignalsSpoilerDamper(snr=-5, channels_spoiled_frac=1.0).fit_transform(clean)
        # Fully attenuated (up to eps), not inverted
        ratio = spoiled[0].to_numpy() / clean[0].to_numpy()
        self.assertTrue(np.all(np.abs(ratio) < 1e-6))

    def test_cubic_unreachable_snr_warns(self):
        with self.assertWarns(UserWarning):
            RawSignalsSpoilerCubicClipper(snr=30, channels_spoiled_frac=1.0).fit_transform(
                make_signals(n=1)
            )

    def test_random_channel_count_includes_all(self):
        spoiler = RawSignalsSpoilerDummy(channels_spoiled_frac=None).fit(make_signals())
        for n_channels, expected in [(1, {1}), (4, {1, 2, 3, 4})]:
            with self.subTest(n_channels=n_channels):
                signal = RawSignal(np.ones((10, n_channels)))
                counts = {
                    len(spoiler._random_channel_selection(signal)) for _ in range(300)
                }
                self.assertEqual(counts, expected)

    def test_sine_time_axis(self):
        signals = RawSignals(sample_rate=1000)
        signals.append(RawSignal(np.full((100, 1), 1e-9)))
        spoiler = RawSignalsSpoilerSine(freq_deviation=0, channels_spoiled_frac=1.0, snr=0)
        spoiler.fit(signals)
        reference_state = RawSignalsSpoilerSine(random_state=spoiler.random_state).fit(signals)
        reference_state._random_state.random(1)
        phase = 4 * np.pi * reference_state._random_state.random(1)[0] - 2 * np.pi

        interference = spoiler.transform(signals)[0].to_numpy()[:, 0] - 1e-9
        expected = np.sin(2 * np.pi * 50 * np.arange(100) / 1000 + phase)
        self.assertTrue(
            np.allclose(
                interference / np.max(np.abs(interference)),
                expected / np.max(np.abs(expected)),
                atol=1e-6,
            )
        )

    def test_multiple_reproducible_and_independent_defaults(self):
        def run():
            return RawSignalsSpoilerMultiple(
                spoilers=[RawSignalsSpoilerGauss()], spoiled_fraction=0.5
            ).fit_transform(make_signals(n=10))

        a, b = run(), run()
        self.assertEqual(list(a.get_labels()), list(b.get_labels()))
        self.assertEqual(a, b)

        m1, m2 = RawSignalsSpoilerMultiple(), RawSignalsSpoilerMultiple()
        m1.fit(make_signals())
        m2.fit(make_signals())
        self.assertIsNot(m1._spoilers[0], m2._spoilers[0])

    def test_no_solver_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for cls in [RawSignalsSpoilerCubicClipper, RawSignalsSpoilerTanhClipper, RawSignalsSpoilerRappAM]:
                cls(snr=10, channels_spoiled_frac=1.0).fit_transform(make_signals(n=1))


if __name__ == "__main__":
    unittest.main()
