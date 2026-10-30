"""Helper code for Milestone 4 (kept in a module so the saved .pkl pipelines can be re-loaded anywhere).

To load a saved model:   import m4_utils, pickle;  model = pickle.load(open("models/randomforest_model.pkl", "rb"))
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRWinsorizer(BaseEstimator, TransformerMixin):
    """Cap values outside the Tukey fences [Q1 - k*IQR, Q3 + k*IQR] instead of deleting rows.

    The fences are learned on the TRAINING data only (inside each cross-validation fold), so no
    information from validation/test rows leaks into the preprocessing. Missing values stay missing
    (the imputer that follows handles them). Rows are never removed.
    """

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)   # NaN stays NaN

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
