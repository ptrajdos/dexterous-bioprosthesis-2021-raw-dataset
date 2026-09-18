from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class EstimatorWithExtractor(BaseEstimator, ClassifierMixin):
    def __init__(self, extractor, estimator) -> None:
        super().__init__()
        self.extractor = extractor
        self.estimator = estimator

    def fit(self, X, y=None):
        """
        Extract features and then fits the estimator!
        """
        X_t, y_t, _ = self.extractor.fit_transform(X, y)
        self.estimator.fit(X_t, y_t)
        self.is_fitted_ = True

        return self

    def fit_predict(self, X, y=None):
        """ """
        return self.fit(X, y).predict(X)

    def _check_is_fitted(self):
        if not hasattr(self, "is_fitted_"):
            raise NotFittedError(
                "This estimator is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

    def predict(self, X):
        """
        Extract features and then predicts with the estimator!
        """
        self._check_is_fitted()

        X_t, y_t, _ = self.extractor.transform(X)
        prediction = self.estimator.predict(X_t)

        return prediction

    def predict_proba(self, X):
        """
        Extract features and then predicts probabilities with the estimator!
        """
        self._check_is_fitted()

        X_t, y_t, _ = self.extractor.transform(X)
        prediction = self.estimator.predict_proba(X_t)

        return prediction

    @property
    def classes_(self):
        return self.estimator.classes_
