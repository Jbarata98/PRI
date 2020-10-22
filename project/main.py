from library import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pympler import asizeof
from itertools import groupby
import xml.etree.ElementTree as ET


import os
import time
import tarfile
import tqdm


COLLECTION_LEN = 807168
COLLECTION_PATH = 'collection/'
DATASET = 'rcv1'
TRAIN_DATE_SPLIT = '19960930'
AVAILABLE_DATA = ('newsitem', 'headline', 'text', 'dateline', 'byline')


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


def parse_xml_doc(tree):
    root = tree.getroot()
    for child in root:
        print(child.text)
    return 0

def tqdm_generator(members, n):
    for member in tqdm.tqdm(members, total=n):
        yield member


def main():
    if os.path.isdir(COLLECTION_PATH + 'rcv1'):
        print(f'Directory "{DATASET}" already exists, moving on to indexing.')
    else:
        print('Directory "rcv1" not found. Extracting "rcv.tar.xz"')
        with tarfile.open('collection/rcv1.tar.xz', 'r:xz') as D:
            D.extractall('collection/', members=tqdm_generator(D, COLLECTION_LEN))

    train_dirs, test_dirs = [list(items) for key, items in groupby(sorted(os.listdir(COLLECTION_PATH + 'rcv1'))[:-3], lambda x: x == TRAIN_DATE_SPLIT) if not key]
    train_dirs.append(TRAIN_DATE_SPLIT)

    for directory in train_dirs[:1]:
        for file_name in os.listdir(f"{COLLECTION_PATH}{DATASET}/{directory}")[:1]:
            print(parse_xml_doc(ET.parse(f"{COLLECTION_PATH}{DATASET}/{directory}/{file_name}")))

    return 0


if __name__ == '__main__':
    main()
