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


class AbstractCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series, y=None) -> pd.Series:
        X = X.str.replace("[^a-zA-Z0-9]+", " ", regex=True)
        X = X.str.replace(" +", " ", regex=True)
        X = X.str.strip()
        X = X.str.lower()
        return X


class StopWordsRemover(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words: list):
        self.stop_words = stop_words

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series, y=None) -> pd.Series:
        return X.apply(lambda text: " ".join(
            [word for word in text.split() if word not in self.stop_words]
        ))

class TextSplitter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series, y=None) -> list:
        return [text.split() for text in X]
