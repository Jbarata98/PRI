import os
import re
import time
import tarfile
import json
import numpy
import nltk
import string

from library import *
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pympler import asizeof
from itertools import groupby
from bs4 import BeautifulSoup
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

COLLECTION_LEN = 807168
COLLECTION_PATH = 'collection/'
DATASET = 'rcv1'
QRELS = 'qrels.test.txt'
TRAIN_DATE_SPLIT = '19960930'
AVAILABLE_DATA = ('headline', 'p', 'dateline', 'byline')
DATA_HEADER = ('newsitem', 'itemid')
TOPICS = "topics.txt"
TOPICS_CONTENT = ('num', 'title', 'desc', 'narr')
DEFAULT_K = 5
BOOLEAN_ROUND_TOLERANCE = 1 - 0.2
OVERRIDE_SAVED_JSON = False

topics = {}
qrels = {}
labeled_docs = []


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


def SimplePreprocessor(article):
    def remove_chars_re(subj, chars):
        return re.sub(u'(?u)[' + re.escape(chars) + ']', ' ', subj)

    return remove_chars_re(article.lower(), string.punctuation)


class InvertedIndex:
    def __init__(self, D, preprocessor=None, tokenizer=None):
        self.test = D['test']
        raw_text_test = self._raw_text_from_dict(self.test)
        self.boolean_index = CountVectorizer(preprocessor=preprocessor, binary=True, tokenizer=tokenizer)
        self.boolean_test_matrix = self.boolean_index.fit_transform(tqdm(raw_text_test, desc=f'{"INDEXING BOOLEAN":20}'))
        self.tfidf_index = TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer, vocabulary=self.boolean_index.vocabulary)
        self.tfidf_test_matrix = self.tfidf_index.fit_transform(tqdm(raw_text_test, desc=f'{"INDEXING TFIDF":20}'))

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
        return [doc_id for doc_id in self.test.keys()]

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


def indexing(D, preprocessor=None, tokenizer=None, *args):
    start_time = time.time()
    # print(json.dumps(train_docs, indent=2))
    I = InvertedIndex(D, preprocessor=preprocessor, tokenizer=tokenizer)
    return I, time.time() - start_time, asizeof.asizeof(I)


def extract_topic_query(q, I: InvertedIndex, k=DEFAULT_K, metric='idf', *args):
    raw_text, term_scores = ' '.join(topics[q].values()), []
    if metric == 'tfidf':
        scores = I.tfidf_transform([raw_text]).todense().A[0]
        term_scores = {term: scores[i] for term, i in I.vocabulary.items() if scores[i] != 0}
    elif metric == 'idf':
        term_scores = {term: I.get_term_idf(term) for term in set(I.build_analyzer()(raw_text))}

    return sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:k]


def boolean_query(q, I: InvertedIndex, k, metric='idf', *args):
    extracted_terms = [' '.join(list(zip(*extract_topic_query(q, I, k, metric, *args)))[0])]
    topic_boolean = I.boolean_transform(extracted_terms)
    dot_product = numpy.dot(topic_boolean, I.boolean_test_matrix.T).A[0]
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


def read_documents(docs, dirs, mode='test', sample_size=None):
    docs[mode] = {}
    for directory in tqdm(dirs, desc=f'{"READING TEST":20}'):
        for file_name in tqdm(sorted(os.listdir(f"{COLLECTION_PATH}{DATASET}/{directory}"))[:sample_size], desc=f'{f"  DIR[{directory}]":20}', leave=False):
            docs['test'].update(parse_xml_doc(f"{COLLECTION_PATH}{DATASET}/{directory}/{file_name}"))


def parse_qrels(filename):
    parsed_qrels, labeled_list = defaultdict(list), []
    with open(filename, encoding='utf8') as f:
        for line in tqdm(f.readlines(), desc=f'{"READING QRELS":20}'):
            q_id, doc_id, relevance = line.split()
            if int(relevance.replace('\n', '')):
                parsed_qrels[q_id].append(doc_id)
            labeled_list.append(doc_id)

    return dict(parsed_qrels), labeled_list


def main():
    global topics
    if os.path.isdir(f'{COLLECTION_PATH}{DATASET}'):
        print(f'Directory "{DATASET}" already exists, moving on to indexing.', flush=True)
    else:
        print('Directory "rcv1" not found. Extracting "rcv.tar.xz"', flush=True)
        with tarfile.open('collection/rcv1.tar.xz', 'r:xz') as D:
            D.extractall('collection/', members=tqdm_generator(D, COLLECTION_LEN))

    if not OVERRIDE_SAVED_JSON and os.path.isfile(f'{COLLECTION_PATH}{DATASET}.json'):
        docs = json.loads(open(f'{COLLECTION_PATH}{DATASET}.json', encoding='ISO-8859-1').read())
    else:
        train_dirs, test_dirs = [list(items) for key, items in groupby(sorted(os.listdir(COLLECTION_PATH + DATASET))[:-3], lambda x: x == TRAIN_DATE_SPLIT) if not key]
        train_dirs.append(TRAIN_DATE_SPLIT)

        docs = {'train': {}, 'test': {}}
        read_documents(docs, test_dirs[:5], 'test', sample_size=1000)
        # read_documents(docs, test_dirs[:5], 'train')

        with open(f'{COLLECTION_PATH}{DATASET}.json', 'w', encoding='ISO-8859-1') as f:
            f.write(json.dumps(docs, indent=4))

    # print(json.dumps(train_docs, indent=2))
    I, indexing_time, indexing_space = indexing(docs, preprocessor=SimplePreprocessor, tokenizer=LemmaTokenizer())
    print(I.get_matrix_data())
    global topics, qrels, labeled_docs
    topics = parse_topics(f"{COLLECTION_PATH}{TOPICS}")
    qrels, labeled_docs = parse_qrels(f"{COLLECTION_PATH}{QRELS}")
    print(f'Indexing time: {indexing_time:10.3f}s, Indexing space: {indexing_space / (1024 ** 2):10.3f}mb')
    for q in topics:
        # print('Topic:', topics[q], sep='\n')
        # print('Topic Keywords:', *extract_topic_query(q, I, k=5, metric='tfidf'), sep='\n')
        doc_ids = boolean_query(q, I, k=10, metric='tfidf')
        # print("Relevant documents:", [I.test[doc_id] for doc_id in doc_ids])
        print(sorted(doc_ids), sorted(qrels[q]), '\n', sep='\n')
    return 0


if __name__ == '__main__':
    main()
