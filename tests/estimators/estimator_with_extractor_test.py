import unittest

from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_mav import (
    NpSignalExtractorMav,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.np_signal_extractors.np_signal_extractor_ssc import (
    NpSignalExtractorSsc,
)
from dexterous_bioprosthesis_2021_raw_datasets.set_creators.set_creator_dwt import (
    SetCreatorDWT,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import (
    RawSignalsCreatorSines,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GridSearchCV

from dexterous_bioprosthesis_2021_raw_datasets.estimators.estimator_with_extractor import (
    EstimatorWithExtractor,
)


def wavelet_extractor2(wavelet_level=2):
    extractor = SetCreatorDWT(
        num_levels=wavelet_level,
        wavelet_name="db6",
        extractors=[
            NpSignalExtractorMav(),
            NpSignalExtractorSsc(),
        ],
    )
    return extractor


class EstimatorWithExtractorTest(unittest.TestCase):

    def get_estimators(self):
        extractor = wavelet_extractor2(wavelet_level=2)
        estimator = RandomForestClassifier()

        params = {
            "estimator__n_estimators": [10, 20],
        }
        gs = GridSearchCV(
            estimator=EstimatorWithExtractor(extractor=extractor, estimator=estimator),
            param_grid=params,
            cv=10,
        )
        return [
            EstimatorWithExtractor(extractor=extractor, estimator=estimator),
            gs,
        ]

    def test_basic(self):

        raw_set = RawSignalsCreatorSines().get_set()
        yr = raw_set.get_labels()

        X, y = raw_set, yr

        for estimator in self.get_estimators():

            estimator.fit(X, y)

            y_pred = estimator.predict(X)
            self.assertIsNotNone(y_pred, "Predictions are none")

            kappa = cohen_kappa_score(y, y_pred)
            self.assertTrue(
                kappa > 0, "Classifier should be better than random guessing"
            )
