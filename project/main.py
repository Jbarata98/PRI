import os
import re
import time
import tarfile
import json
import numpy as np
import nltk
import string
import random
import matplotlib.pyplot as plt
from sklearn.metrics import dcg_score, ndcg_score, average_precision_score

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve

import ml_metrics
from pympler import asizeof
from itertools import groupby
from bs4 import BeautifulSoup
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from whoosh import index
from whoosh.fields import *
from whoosh.qparser import *
from whoosh.analysis import *

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
OVERRIDE_SUBSET_JSON = False
USE_ONLY_EVAL = True
MaxMRRRank = 10
EVAL_SAMPLE_SIZE = 3000

topics = {}
topic_index = {}
doc_index = []

def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend, loc='best', fancybox=True, shadow=True, borderaxespad=0)


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
        self.test = D
        # raw_text_test = self._raw_text_from_dict(self.test)
        # self.boolean_index = CountVectorizer(preprocessor=preprocessor, binary=True, tokenizer=tokenizer)
        # self.boolean_test_matrix = self.boolean_index.fit_transform(tqdm(raw_text_test, desc=f'{"INDEXING BOOLEAN":20}'))
        # self.tfidf_index = TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer, vocabulary=self.boolean_index.vocabulary)
        # self.tfidf_test_matrix = self.tfidf_index.fit_transform(tqdm(raw_text_test, desc=f'{"INDEXING TFIDF":20}'))
        self._save_index(self.test)

    @staticmethod
    def _raw_text_from_dict(doc_dict):
        return [' '.join(list(doc.values())) for doc in doc_dict.values()]  # Joins all docs in a single list of raw_doc strings

    @staticmethod
    def _save_index(D):
        if not os.path.exists("whoosh"):
            os.mkdir("whoosh")
        analyzer = StemmingAnalyzer()
        schema = Schema(id=ID(stored=True, unique=True), **{tag: TEXT(phrase=False, analyzer=analyzer) for tag in AVAILABLE_DATA})  # Schema
        ix = index.create_in("whoosh", schema)
        writer = ix.writer()
        for doc_id, doc in tqdm(D.items(), desc=f'{"INDEXING WHOOSH":20}'):
            writer.update_document(id=doc_id, **{tag: doc[tag] for tag in AVAILABLE_DATA})
        writer.commit()

    @property
    def idf(self):
        return self.tfidf_index.idf_

    @property
    def vocabulary(self):
        return self.boolean_index.vocabulary_

    @property
    def doc_ids(self):
        return [doc_id for doc_id in self.test.keys()]

    def search_index(self, string, k=10):
        id_list = []
        ix = index.open_dir("whoosh")
        with ix.searcher() as searcher:
            q = MultifieldParser(AVAILABLE_DATA, ix.schema, group=OrGroup)
            q.remove_plugin_class(PhrasePlugin)
            q = q.parse(string)
            results = searcher.search(q, limit=k)
            for r in results:
                id_list.append((r['id'], r.score))
        return id_list

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
    print(extracted_terms)
    dot_product = np.dot(topic_boolean, I.boolean_test_matrix.T).A[0]
    return [doc_id for i, doc_id in enumerate(I.doc_ids) if dot_product[i] >= round(BOOLEAN_ROUND_TOLERANCE * k)]


def ranking(q, p, I: InvertedIndex, *args):
    return I.search_index(' '.join(topics[q].values()), p)


def calc_precision_based_measures(predicted_ids, expected_ids, nr_documents=10, metric=None):
    def precision(_predicted, _expected):
        return len(set(_predicted).intersection(set(_expected))) / len(_predicted)

    def recall(_predicted, _expected):
        return len(set(_predicted).intersection(set(_expected))) / len(_expected)

    def f1(_predicted, _expected):
        pre, rec = precision(_predicted, _expected), recall(_predicted, _expected)
        return 0.0 if pre == rec == 0 else 2 * pre * rec / (pre + rec)

    def map(_predicted, _expected):
        pre, rec = precision(_predicted, _expected), recall(_predicted, _expected)
        return 0.0 if pre == rec == 0 else ml_metrics.mapk([_expected], [_predicted], nr_documents)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map': map,
    }

    if metric is None:
        metric = metrics.keys()
    return {measure: metrics[measure]([int(i) for i in predicted_ids], [int(i) for i in expected_ids]) for measure in metric}



def precision_recall_generator(predicted, expected):
    tp = 0
    for i, id in enumerate(predicted):
        if id in expected:
            tp += 1
        yield tp / (i + 1), tp / max(1, len(expected))



''' Code Adapted from https://gist.github.com/bwhite/3726239'''
def calc_gain_based_measures(scores, nr_documents=10, metric=None):
    def dcg(scores, k):
        r = np.asfarray(scores)[:k]
        if r.size:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        return 0

    def ndcg(scores, k):
        dcg_max = dcg(sorted(scores, reverse=True), k)
        if not dcg_max:
            return 0.
        return dcg(scores, k) / dcg_max

    metrics = {
        'dcg': dcg,
        'ndcg': ndcg
    }

    if metric is None:
        metric = metrics.keys()
    return {measure: metrics[measure](scores, nr_documents) for measure in
            metric}


def MRR(predicted, expected, metric=None):
    MRR = 0
    for i, qid in zip(range(MaxMRRRank), predicted):
        if qid in expected:
            MRR += 1 / (i + 1)
            break
    return {'MRR': MRR}


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


def read_documents(dirs, sample_size=None):
    docs = {}
    for directory in tqdm(dirs, desc=f'{"PARSING DATASET":20}'):
        for file_name in tqdm(sorted(os.listdir(f"{COLLECTION_PATH}{DATASET}/{directory}"))[:sample_size], desc=f'{f"  DIR[{directory}]":20}', leave=False):
            docs.update(parse_xml_doc(f"{COLLECTION_PATH}{DATASET}/{directory}/{file_name}"))
    return docs


def parse_qrels(filename):
    topic_index, doc_index = defaultdict(list), defaultdict(list)
    with open(filename, encoding='utf8') as f:
        for line in tqdm(f.readlines(), desc=f'{"READING QRELS":20}'):
            q_id, doc_id, relevance = line.split()
            if int(relevance.replace('\n', '')):
                topic_index[q_id].append(doc_id)
                doc_index[doc_id].append(q_id)

    return dict(topic_index), dict(doc_index)


def get_subset(adict, subset):
    return {key: adict[key] for key in subset if key in adict}


def main():
    global topics, topic_index, doc_index

    extract_dataset()

    topics = parse_topics(f"{COLLECTION_PATH}{TOPICS}")
    topic_index, doc_index = parse_qrels(f"{COLLECTION_PATH}{QRELS}")

    if USE_ONLY_EVAL and os.path.isfile(f'{COLLECTION_PATH}{DATASET}_sub.json'):
        print(f"{DATASET}_sub.json found, loading it...")
        docs = json.loads(open(f'{COLLECTION_PATH}{DATASET}.json', encoding='ISO-8859-1').read())
    else:
        print(f"Loading full dataset...")
        if not OVERRIDE_SAVED_JSON and os.path.isfile(f'{COLLECTION_PATH}{DATASET}.json'):
            print(f"{DATASET}.json found, loading it...")
            docs = json.loads(open(f'{COLLECTION_PATH}{DATASET}.json', encoding='ISO-8859-1').read())
        else:
            print(f"{DATASET}.json not found, parsing full dataset...")
            docs = parse_dataset()

        if USE_ONLY_EVAL:
            docs = get_subset(docs, doc_index)
            print(f"Saving eval set to {DATASET}_sub.json...")
            with open(f'{COLLECTION_PATH}{DATASET}_sub.json', 'w', encoding='ISO-8859-1') as f:
                f.write(json.dumps(docs, indent=4))

    sampled_doc_ids = random.sample(doc_index.keys(), EVAL_SAMPLE_SIZE)
    doc_index = get_subset(doc_index, sampled_doc_ids)
    topic_index = defaultdict(list)
    for doc_id, doc_topics in doc_index.items():
        for doc_topics in doc_topics:
            topic_index[doc_topics].append(doc_id)
    docs = get_subset(docs, doc_index)


    # I, indexing_time, indexing_space = indexing(docs, preprocessor=SimplePreprocessor, tokenizer=LemmaTokenizer())
    I, indexing_time, indexing_space = indexing(docs, preprocessor=None, tokenizer=None)

    print(f'Indexing time: {indexing_time:10.3f}s, Indexing space: {indexing_space / (1024 ** 2):10.3f}mb')

    for q in tqdm(topics, desc=f'{f"BOOLEAN RETRIEVAL":20}'):
        # print("Topic:", q)
        # print('Topic:', topics[q], sep='\n')
        # print('Topic Keywords:', *extract_topic_query(q, I, k=5, metric='tfidf'), sep='\n')
        # doc_ids = boolean_query(q, I, k=5, metric='tfidf')
        # print("Relevant documents:", [I.test[doc_id] for doc_id in doc_ids])
        pass

    for q in tqdm(topics, desc=f'{f"RANKING":20}'):
        doc_ids = ranking(q, 100, I)
        print("Predicted:", sorted(doc_ids),
              "Expected:", sorted(topic_index[q]),
              "Precision Measures:", calc_precision_based_measures([ids[0] for ids in doc_ids], topic_index[q], nr_documents=MaxMRRRank),
              "Gain Measures:", calc_gain_based_measures([scores[1] for scores in doc_ids], nr_documents=MaxMRRRank),
              '\n', sep='\n')
        pr = precision_recall_generator([ids[0] for ids in doc_ids], topic_index[q])
        Y = {}
        Y['precision'], recall = zip(*[[p, r] for (p, r) in pr])
        multiple_line_chart(plt.gca(), recall, Y, 'Precision-Recall Curve', 'recall', 'precision')
        plt.show()

    return 0


def parse_dataset():
    train_dirs, test_dirs = [list(items) for key, items in groupby(sorted(os.listdir(COLLECTION_PATH + DATASET))[:-3], lambda x: x == TRAIN_DATE_SPLIT) if not key]
    train_dirs.append(TRAIN_DATE_SPLIT)
    docs = read_documents(test_dirs[:], sample_size=None)

    print(f"Saving full set to {DATASET}.json...")
    with open(f'{COLLECTION_PATH}{DATASET}.json', 'w', encoding='ISO-8859-1') as f:
        f.write(json.dumps(docs, indent=4))
    return docs


def extract_dataset():
    if os.path.isdir(f'{COLLECTION_PATH}{DATASET}'):
        print(f'Directory "{DATASET}" already exists, no extraction needed.', flush=True)
    else:
        print('Directory "rcv1" not found. Extracting "rcv.tar.xz..."', flush=True)
        with tarfile.open('collection/rcv1.tar.xz', 'r:xz') as D:
            D.extractall('collection/', members=tqdm_generator(D, COLLECTION_LEN))


if __name__ == '__main__':
    main()
