import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingAbstractReplacer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None) -> BaseEstimator:
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.Series:
        return pd.Series(np.where(
            X["AbstractNarration"].isnull(),
            X["AwardTitle"],
            X["AbstractNarration"],
        ))


class XMLRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.str.replace("(&lt;br/&gt;)+", " ", regex=True)
        return X
