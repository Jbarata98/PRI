import os
import time
import tarfile
import json
import numpy

from library import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pympler import asizeof
from itertools import groupby
from bs4 import BeautifulSoup
from tqdm import tqdm

COLLECTION_LEN = 807168
COLLECTION_PATH = 'collection/'
DATASET = 'rcv1'
TRAIN_DATE_SPLIT = '19960930'
AVAILABLE_DATA = ('headline', 'p', 'dateline', 'byline')
DATA_HEADER = ('newsitem', 'itemid')
TOPICS = "topics.txt"
TOPICS_CONTENT = ('num', 'title', 'desc', 'narr')
DEFAULT_K = 5
BOOLEAN_ROUND_TOLERANCE = 1 - 0.2

topics = {}


class InvertedIndex:
    def __init__(self, D, preprocessor=None):
        raw_text_list = self._raw_text_from_dict(D)
        self.D = D
        self.boolean_index = CountVectorizer(preprocessor=preprocessor, binary=True)
        self.boolean_matrix = self.boolean_index.fit_transform(raw_text_list)
        self.tfidf_index = TfidfVectorizer(preprocessor=preprocessor, vocabulary=self.boolean_index.vocabulary)
        self.tfidf_matrix = self.tfidf_index.fit_transform(raw_text_list)

    @staticmethod
    def _raw_text_from_dict(doc_dict):
        return [' '.join(list(doc.values())) for doc in doc_dict.values()]  # Joins all docs in a single list of raw_doc strings

    @property
    def idf(self):
        return self.tfidf_index.idf_

    @property
    def vocabulary(self):
        return self.boolean_index.vocabulary_

    @property
    def doc_ids(self):
        return [doc_id for doc_id in self.D.keys()]

    def get_term_idf(self, term):
        return 0 if term not in self.vocabulary else self.idf[self.vocabulary[term]]

    def get_matrix_data(self):
        return {'len': len(self.idf), 'vocabulary': self.vocabulary, 'idf': self.idf}

    def boolean_transform(self, raw_documents):
        return self.boolean_index.transform(raw_documents)

    def tfidf_transform(self, raw_documents):
        return self.tfidf_index.transform(raw_documents)

    def build_analyzer(self):
        return self.tfidf_index.build_analyzer()


def indexing(D, preprocess=None, *args):
    start_time = time.time()
    # print(json.dumps(train_docs, indent=2))
    I = InvertedIndex(D, preprocessor=preprocess)
    return I, time.time() - start_time, asizeof.asizeof(I)


def extract_topic_query(q, I: InvertedIndex, k=DEFAULT_K, metric='idf', *args):
    raw_text = ' '.join(topics[q].values())
    if metric == 'tfidf':
        scores = I.tfidf_transform([raw_text]).todense().A[0]
        term_scores = {term: scores[i] for term, i in I.vocabulary.items() if scores[i] != 0}
    elif metric == 'idf':
        term_scores = {term: I.get_term_idf(term) for term in set(I.build_analyzer()(raw_text))}

    return sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:k]


def boolean_query(q, I: InvertedIndex, k, metric='idf', *args):
    extracted_terms = [' '.join(list(zip(*extract_topic_query(q, I, k, metric, *args)))[0])]
    topic_boolean = I.boolean_transform(extracted_terms)
    dot_product = numpy.dot(topic_boolean.A, I.boolean_matrix.A.T)[0]
    return [doc_id for i, doc_id in enumerate(I.doc_ids) if dot_product[i] >= round(BOOLEAN_ROUND_TOLERANCE * k)]


def parse_xml_doc(filename):
    parsed_doc = {}
    with open(filename, encoding='ISO-8859-1') as f:
        soup = BeautifulSoup(f.read().strip(), 'lxml')

        for tag in AVAILABLE_DATA:
            parsed_doc[tag] = ''.join([str(e.string) for e in soup.find_all(tag)])

        return {soup.find(DATA_HEADER[0])[DATA_HEADER[1]]: parsed_doc}


def parse_topics(filename):
    parsed_topics = {}
    with open(filename, encoding='ISO-8859-1') as f:
        soup = BeautifulSoup(f.read().strip(), 'lxml')
        for topic in soup.find_all('top'):
            parsed_topics.update(parse_topic(topic.get_text()))
    return parsed_topics


def parse_topic(topic):
    topic_dict = {}
    contents = topic.split(':')
    id_title = list(filter(None, contents[1].split('\n')))
    topic_dict['title'] = id_title[1].strip()
    topic_dict['desc'] = ''.join(contents[2].split('\n')[:-1]).strip()
    topic_dict['narr'] = ''.join(contents[-1].split('\n')).strip()
    return {id_title[0].strip(): topic_dict}


def tqdm_generator(members, n):
    for member in tqdm(members, total=n):
        yield member


def main():
    global topics
    if os.path.isdir(COLLECTION_PATH + 'rcv1'):
        print(f'Directory "{DATASET}" already exists, moving on to indexing.', flush=True)
    else:
        print('Directory "rcv1" not found. Extracting "rcv.tar.xz"', flush=True)
        with tarfile.open('collection/rcv1.tar.xz', 'r:xz') as D:
            D.extractall('collection/', members=tqdm_generator(D, COLLECTION_LEN))

    train_dirs, test_dirs = [list(items) for key, items in groupby(sorted(os.listdir(COLLECTION_PATH + DATASET))[:-3], lambda x: x == TRAIN_DATE_SPLIT) if not key]
    train_dirs.append(TRAIN_DATE_SPLIT)

    train_docs = {}
    for directory in tqdm(train_dirs[:], desc=f'{"INDEXING":15}'):
        for file_name in tqdm(sorted(os.listdir(f"{COLLECTION_PATH}{DATASET}/{directory}"))[:10], desc=f'{f"  DIR[{directory}]":15}', leave=False):
            train_docs.update(parse_xml_doc(f"{COLLECTION_PATH}{DATASET}/{directory}/{file_name}"))

    # print(json.dumps(train_docs, indent=2))
    I, indexing_time, indexing_space = indexing(train_docs)
    # print(I.get_matrix_data())
    global topics
    topics = parse_topics(f"{COLLECTION_PATH}{TOPICS}")
    print(f'Indexing time: {indexing_time:10.3f}s, Indexing space: {indexing_space / (1024 ** 2):10.3f}mb')
    print(I.doc_ids)
    for q in topics:
        print('Topic:', topics[q], sep='\n')
        print('Topic Keywords:', *extract_topic_query(q, I, k=5, metric='tfidf'), sep='\n')
        doc_ids = boolean_query(q, I, k=5, metric='tfidf')
        print("Relevant documents:", [I.D[doc_id] for doc_id in doc_ids])
    return 0


if __name__ == '__main__':
    main()
