from library import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pympler import asizeof
from itertools import groupby
from bs4 import BeautifulSoup
from tqdm import tqdm

import os
import time
import tarfile
import json

COLLECTION_LEN = 807168
COLLECTION_PATH = 'collection/'
DATASET = 'rcv1'
TRAIN_DATE_SPLIT = '19960930'
AVAILABLE_DATA = ('headline', 'p', 'dateline', 'byline')


class InvertedIndex:
    def __init__(self, D, preprocessor=None):
        raw_text_list = self._raw_text_from_dict(D)
        self.boolean_index = CountVectorizer(preprocessor=preprocessor, binary=True)
        self.boolean_matrix = self.boolean_index.fit_transform(raw_text_list)
        self.tfidf_index = TfidfVectorizer(preprocessor=preprocessor, vocabulary=self.boolean_index.vocabulary)
        self.tfidf_matrix = self.tfidf_index.fit_transform(raw_text_list)

    @staticmethod
    def _raw_text_from_dict(doc_dict):
        return [' '.join(list(doc.values())) for docs in doc_dict.values() for doc in docs] #Joins all docs in a single list of raw_doc strings

    @property
    def idf(self):
        return self.tfidf_index.idf_
    @property
    def vocabulary(self):
        return self.boolean_index.vocabulary_

    def get_matrix_data(self):
        return f'{self.idf}\n\n{self.vocabulary}'


def indexing(D, preprocess=None, *args):
    I, start_time = InvertedIndex(preprocessor=preprocess), time.time()
    return I.fit(D).idf, start_time - time.time(), asizeof.asizeof(I)


def parse_xml_doc(filename):
    parsed_doc = {}

    soup = BeautifulSoup(open(filename).read(), 'lxml')

    for tag in AVAILABLE_DATA:
        parsed_doc[tag] = ''.join([str(e.string) for e in soup.find_all(tag)])

    return parsed_doc


def tqdm_generator(members, n):
    for member in tqdm(members, total=n):
        yield member


def main():
    if os.path.isdir(COLLECTION_PATH + 'rcv1'):
        print(f'Directory "{DATASET}" already exists, moving on to indexing.', flush=True)
    else:
        print('Directory "rcv1" not found. Extracting "rcv.tar.xz"', flush=True)
        with tarfile.open('collection/rcv1.tar.xz', 'r:xz') as D:
            D.extractall('collection/', members=tqdm_generator(D, COLLECTION_LEN))

    train_dirs, test_dirs = [list(items) for key, items in groupby(sorted(os.listdir(COLLECTION_PATH + 'rcv1'))[:-3], lambda x: x == TRAIN_DATE_SPLIT) if not key]
    train_dirs.append(TRAIN_DATE_SPLIT)

    train_docs = {}
    for directory in train_dirs[:1]:
        train_docs[directory] = []
        for file_name in tqdm(sorted(os.listdir(f"{COLLECTION_PATH}{DATASET}/{directory}"))[:10], desc=f'dir:{directory}'):
            train_docs[directory].append(parse_xml_doc(f"{COLLECTION_PATH}{DATASET}/{directory}/{file_name}"))

    # print(json.dumps(train_docs, indent=2))
    I = InvertedIndex(train_docs)
    print(I.get_matrix_data())
    return 0


if __name__ == '__main__':
    main()
