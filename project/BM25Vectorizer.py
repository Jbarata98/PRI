""" Implementation of OKapi BM25 with sklearn's TfidfVectorizer
Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)
ADAPTED
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy import sparse


class BM25Vectorizer(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1
        self.full = True

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        self.y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = self.y.sum(1).mean()
        self.X = X

    def transform_query(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]

        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_ - 1.
        numer = sparse.csr_matrix(X.multiply(np.broadcast_to(idf, X.shape))) * (k1 + 1)

        return (numer / denom) if self.full else (numer / denom)[:, q.indices]

    def transform(self, X):
        return np.array([self.transform_query(x, [x]).A1 for x in X])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class BM25Scorer(BM25Vectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.full = False

    def transform(self, X):
        return normalize(np.array([self.transform_query(x, self.X).sum(1).A1 for x in X]))