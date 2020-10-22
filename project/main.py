from library import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pympler import asizeof
import os
import time


class InvertedIndex:
    def __init__(self, preprocessor=None):
        self.boolean_index = CountVectorizer(input='filename', preprocessor=preprocessor, binary=True)
        self.tfidf_index = TfidfVectorizer(input='filename', preprocessor=preprocessor)
        self.tfidf_matrix = None
        self.boolean_matrix = None

    @property
    def idf(self):
        return self.tfidf_index.idf_

    def fit(self, D):
        document_collection = [D + '/' + filename for filename in os.listdir(D)]

        self.tfidf_matrix = self.tfidf_index.fit_transform(document_collection)
        self.boolean_matrix = self.boolean_index.fit_transform(document_collection)
        return self


def indexing(D, preprocess=None, *args):
    I, start_time = InvertedIndex(preprocessor=preprocess), time.time()
    return I.fit(D).idf, start_time - time.time(), asizeof.asizeof(I)


def main():
    return 0


if __name__ == '__main__':
    main()
