import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class TextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str ="all-MiniLM-L12-v2"):
        """
        Class that vectorizes text using BERT.

        Parameters
        ----------
        model_name: str
            Name of the sentence-transformer model to use. Default is "all-MiniLM-L12-v2".
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def fit(self, X: list, y=None) -> BaseEstimator:
        return self

    def transform(self, X: list) -> np.ndarray:
        """
        Vectorizes text using BERT.

        Parameters
        ----------
        X: list
            List of strings to vectorize.

        Returns
        -------
        numpy.ndarray
            Array of BERT vectors from pre-trained BERT model.
        """
        X = self.model.encode(X, show_progress_bar=True)
        return X
