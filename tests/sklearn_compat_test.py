"""scikit-learn compatibility of raw signal filters, augmenters and spoilers.

All concrete classes are discovered automatically. Each one is checked for:
- estimator tags (scikit-learn >= 1.6),
- __sklearn_is_fitted__ before and after fitting,
- working as a step of a scikit-learn Pipeline. In scikit-learn >= 1.6
  Pipeline.transform checks that its last step is fitted, which needs both above.
"""
import importlib
import inspect
import logging
import pkgutil
import random
import unittest
import warnings

import numpy as np
from sklearn.pipeline import Pipeline

import dexterous_bioprosthesis_2021_raw_datasets as package
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter import (
    RawSignalsFilter,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_filters.raw_signals_filter_rectifier import (
    RawSignalsFilterRectifier,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_spoilers.raw_signals_spoiler_interface import (
    RawSignalsSpoilerInterface,
)
from tests.testing_tools import generate_sample_data

BASE_CLASSES = (RawSignalsFilter, RawSignalsAugumenter, RawSignalsSpoilerInterface)

# Constructor arguments for classes without a default constructor
CONSTRUCTOR_KWARGS = {
    "RawSignalsFilterChannelIdx": dict(indices_list=[0, 1]),
    "RawSignalsFilterClassRegex": dict(pattern=".*"),
    "RawSignalsFilterClassValue": dict(allowed_values=[0, 1]),
    "RawSignalsFilterColumnNameRegex": dict(pattern="^C[01]"),
    "RawSignalsFilterColumnToClassIdx": dict(indices_list=[0]),
    "RawSignalsFilterColumnToClassRegex": dict(pattern="^C0"),
    "RawSignalsFilterConditionalColumn": dict(
        column_filter_pairs=("^C0", RawSignalsFilterRectifier())
    ),
    "RawSignalsFilterRowIdx": dict(indices_list=[0, 1, 2]),
    "RawSignalsFilterRowSlice": dict(row_slice=slice(0, 50)),
    "RawSignalsFilterWindowSegmentation": dict(window_length=0.5, overlap=0.5),
    "RawSignalsFilterWindowSegmentationFS": dict(window_length=50, overlap=25),  # ms
    "RawSignalsFilterWindowSingle": dict(start_sample=0, end_sample=50),
    "RawSignalsFilterWindowSingleIF": dict(offset=0.1, length=0.5),
    "RawSignalsFilterWindowSingleMs": dict(offset_ms=10, length_ms=50),
}

# Classes whose transform needs an optional dependency
OPTIONAL_DEPENDENCIES = {
    "RawSignalsAugumenterPitchShift": "audiomentations",
    "RawSignalsAugumenterTimeStretch": "audiomentations",
}


def missing_dependency(cls):
    """Return the name of a missing optional dependency of cls, or None."""
    dependency = OPTIONAL_DEPENDENCIES.get(cls.__name__)
    if dependency is None:
        return None
    try:
        importlib.import_module(dependency)
    except ImportError:
        return dependency
    return None


def find_classes(base_classes=BASE_CLASSES):
    """Return all concrete subclasses of base_classes defined in the package."""
    found = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)  # optional dependencies warn on import
        try:
            for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    module = importlib.import_module(module_info.name)
                except ImportError:
                    continue
                for name, cls in inspect.getmembers(module, inspect.isclass):
                    if (
                        cls.__module__ == module.__name__
                        and issubclass(cls, base_classes)
                        and not inspect.isabstract(cls)
                    ):
                        found[name] = cls
        finally:
            logging.disable(logging.NOTSET)
    return found


def create(cls):
    return cls(**CONSTRUCTOR_KWARGS.get(cls.__name__, {}))


class SklearnCompatTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classes = find_classes()

    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        self.raw_set = generate_sample_data(
            signal_number=10, column_number=3, samples_number=200, class_indices=[0, 1]
        )
        self.y = self.raw_set.get_labels()

    def test_classes_found(self):
        for base in BASE_CLASSES:
            with self.subTest(base=base.__name__):
                self.assertTrue(
                    any(issubclass(cls, base) for cls in self.classes.values()),
                    f"No concrete subclasses of {base.__name__} found",
                )

    def test_all_classes_constructible(self):
        for name, cls in self.classes.items():
            with self.subTest(cls=name):
                try:
                    create(cls)
                except TypeError as ex:
                    self.fail(f"Add constructor arguments to CONSTRUCTOR_KWARGS: {ex}")

    def test_sklearn_tags(self):
        try:
            from sklearn.utils import get_tags
        except ImportError:
            self.skipTest("Estimator tags API requires scikit-learn >= 1.6")

        for name, cls in self.classes.items():
            with self.subTest(cls=name):
                tags = get_tags(create(cls))
                self.assertTrue(tags.requires_fit, "Should require fit")
                self.assertIsNotNone(tags.transformer_tags, "Should be a transformer")

    def test_sklearn_is_fitted(self):
        for name, cls in self.classes.items():
            with self.subTest(cls=name):
                obj = create(cls)
                self.assertFalse(obj.__sklearn_is_fitted__(), "Should not be fitted")

                returned = obj.fit(self.raw_set, self.y)
                self.assertIs(returned, obj, "fit should return self")
                self.assertTrue(obj.__sklearn_is_fitted__(), "Should be fitted")

    def test_pipeline_step(self):
        for name, cls in self.classes.items():
            if missing_dependency(cls) is not None:
                continue  # skipTest here would skip the remaining classes too

            with self.subTest(cls=name):
                pipeline = Pipeline([("step", create(cls))])

                pipeline.fit(self.raw_set, self.y)
                transformed = pipeline.transform(self.raw_set)
                self.assertIsNotNone(transformed, "Transform returned None")

                transformed = pipeline.fit_transform(self.raw_set, self.y)
                self.assertIsNotNone(transformed, "fit_transform returned None")

    def test_pipeline_not_fitted(self):
        """Pipeline.transform on an unfitted last step raises NotFittedError (scikit-learn >= 1.6)."""
        try:
            from sklearn.utils import get_tags  # noqa: F401
        except ImportError:
            self.skipTest("Pipeline fitted check requires scikit-learn >= 1.6")

        from sklearn.exceptions import NotFittedError

        for name, cls in self.classes.items():
            with self.subTest(cls=name):
                pipeline = Pipeline([("step", create(cls))])
                with self.assertRaises(NotFittedError):
                    pipeline.transform(self.raw_set)


if __name__ == "__main__":
    unittest.main()
