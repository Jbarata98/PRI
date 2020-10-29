import os
import re
import time
import tarfile
import json
import jsonpickle
import numpy as np
import nltk
import string
import random
import matplotlib.pyplot as plt
import pandas as pd
import ml_metrics
import whoosh.scoring

from json import JSONEncoder
from sklearn.metrics import dcg_score, ndcg_score, average_precision_score
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
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
from metrics import *
from parsers import *

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
BETA = 0.5
K1_TEST_VALS = np.arange(0, 4.1, 0.5)
B_TEST_VALS = np.arange(0, 1.1, 0.2)
DEFAULT_P = 1000
K_TESTS = (1, 3, 5, 10, 20, 50, 100, 200, 500, DEFAULT_P)
FOLDS = 0
RANDOM_STATE = 420

topics = {}
topic_index = {}
doc_index = []
topic_index_n = {}
doc_index_n = []


class NamedBM25F(whoosh.scoring.BM25F):
    def __init__(self, *args, **aargs):
        super().__init__(*args, **aargs)
        self.name = f"BM25_k1_{self.K1:.2f}_b_{self.B:.2f}".replace('.', ',')

    def __str__(self):
        return self.name


class NamedTF_IDF(whoosh.scoring.TF_IDF):
    def __init__(self, *args, **aargs):
        super().__init__(*args, **aargs)
        self.name = f"TF_IDF"

    def __str__(self):
        return self.name


class NamedAnalyzer():
    def __init__(self, analyzer, name):
        self.analyzer = analyzer
        self.name = name

    def __call__(self, *args, **aargs):
        return self.analyzer(*args, **aargs)

    def process_raw_texts(self, raw_texts):
        return [' '.join([token.text for token in self.analyzer(raw_text)]) for raw_text in raw_texts]

    def __repr__(self):
        return repr(self.analyzer)

    def __str__(self):
        return self.name


class LemmaFilter(Filter):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, tokens):
        for t in tokens:
            t.text = self.wnl.lemmatize(t.text)
            yield t


def SimplePreprocessor(article):
    def remove_chars_re(subj, chars):
        return re.sub(u'(?u)[' + re.escape(chars) + ']', ' ', subj)

    return remove_chars_re(article.lower(), string.punctuation)


class InvertedIndex:
    def __init__(self, D, analyzer: NamedAnalyzer = None, scoring=None):
        self.D_name, self.D = D
        self.analyzer = analyzer if analyzer else NamedAnalyzer(StemmingAnalyzer(), "stemming_stopwords")
        self.whoosh_dir = f"whoosh/{self.D_name}_{self.analyzer}"
        # raw_text_test = self._raw_text_from_dict(self.test)
        # self.boolean_index = CountVectorizer(preprocessor=preprocessor, binary=True, tokenizer=tokenizer)
        # self.boolean_test_matrix = self.boolean_index.fit_transform(tqdm(raw_text_test, desc=f'{"INDEXING BOOLEAN":20}'))
        # self.tfidf_index = TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer, vocabulary=self.boolean_index.vocabulary)
        # self.tfidf_test_matrix = self.tfidf_index.fit_transform(tqdm(raw_text_test, desc=f'{"INDEXING TFIDF":20}'))
        self._save_index()
        self.scoring = scoring if scoring else NamedBM25F()
        print(scoring)

    @staticmethod
    def _raw_text_from_dict(doc_dict):
        return [' '.join(list(doc.values())) for doc in
                doc_dict.values()]  # Joins all docs in a single list of raw_doc strings

    def _save_index(self):
        if not os.path.exists("whoosh"):
            os.mkdir("whoosh")
        if not os.path.exists(self.whoosh_dir):
            os.mkdir(self.whoosh_dir)
        if index.exists_in(self.whoosh_dir):
            print(f"Whoosh index found in \"{self.whoosh_dir}\"")
        else:
            print(f"Whoosh index not found, creating in \"{self.whoosh_dir}\"...")
            schema = Schema(id=ID(stored=True, unique=True),
                            **{tag: TEXT(phrase=False, analyzer=self.analyzer) for tag in AVAILABLE_DATA})  # Schema
            ix = index.create_in(self.whoosh_dir, schema)
            writer = ix.writer()
            for doc_id, doc in tqdm(self.D.items(), desc=f'{"INDEXING WHOOSH":20}'):
                writer.add_document(id=doc_id, **{tag: doc[tag] for tag in AVAILABLE_DATA})
            writer.commit()

    @property
    def scoring(self):
        return self.__scoring

    @scoring.setter
    def scoring(self, new_scoring):
        self.__scoring = new_scoring

    @property
    def idf(self):
        return self.tfidf_index.idf_

    @property
    def vocabulary(self):
        return self.boolean_index.vocabulary_

    @property
    def doc_ids(self):
        return [doc_id for doc_id in self.D.keys()]

    def search_index(self, string, k=10):
        id_list = []
        ix = index.open_dir(self.whoosh_dir)
        with ix.searcher(weighting=self.__scoring) as searcher:
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


stem_analyzer = NamedAnalyzer(StemmingAnalyzer(), "stemming_stopwords")
lemma_analyzer = NamedAnalyzer(RegexTokenizer() | LowercaseFilter() | StopFilter() | LemmaFilter(), "lemma_stopwords")


def rprint(x, *args, **pargs):
    print(x, *args, **pargs)
    return x


def extract_topic_query(q, I: InvertedIndex, k=DEFAULT_K, metric='idf', *args):
    raw_text, term_scores = ' '.join(topics[q].values()), []
    if metric == 'tfidf':
        scores = I.tfidf_transform([raw_text]).todense().A[0]
        term_scores = {term: scores[i] for term, i in I.vocabulary.items() if scores[i] != 0}
    elif metric == 'idf':
        term_scores = {term: I.get_term_idf(term) for term in set(I.build_analyzer()(raw_text))}

    return sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:k]


def indexing(D, *args, **aargs):
    start_time = time.time()
    # print(json.dumps(train_docs, indent=2))
    I = InvertedIndex(D, *args, **aargs)
    return I, time.time() - start_time, asizeof.asizeof(I)


def boolean_query(q, I: InvertedIndex, k, metric='idf', *args):
    extracted_terms = [' '.join(list(zip(*extract_topic_query(q, I, k, metric, *args)))[0])]
    topic_boolean = I.boolean_transform(extracted_terms)
    print(extracted_terms)
    dot_product = np.dot(topic_boolean, I.boolean_test_matrix.T).A[0]
    return [doc_id for i, doc_id in enumerate(I.doc_ids) if dot_product[i] >= round(BOOLEAN_ROUND_TOLERANCE * k)]


def ranking(q, p, I: InvertedIndex, *args):
    return I.search_index(' '.join(topics[q].values()), p)


def parse_qrels(filename):
    topic_index, doc_index, topic_index_n, doc_index_n = defaultdict(list), defaultdict(list), defaultdict(
        list), defaultdict(list)
    with open(filename, encoding='utf8') as f:
        for line in tqdm(f.readlines(), desc=f'{"READING QRELS":20}'):
            q_id, doc_id, relevance = line.split()
            if int(relevance.replace('\n', '')):
                topic_index[q_id].append(doc_id)
                doc_index[doc_id].append(q_id)
            else:
                topic_index_n[q_id].append(doc_id)
                doc_index_n[doc_id].append(q_id)

    return dict(topic_index), dict(doc_index), dict(topic_index_n), dict(doc_index_n)


def get_subset(adict, subset):
    return {key: adict[key] for key in subset if key in adict}


def main():
    global topics, topic_index, doc_index, topic_index_n, doc_index_n

    # EXTRACTION
    extract_dataset()

    # <Build Q>
    topics = parse_topics(f"{COLLECTION_PATH}{TOPICS}")
    topic_index, doc_index, topic_index_n, doc_index_n = parse_qrels(f"{COLLECTION_PATH}{QRELS}")

    # </Build Q>

    # <Dataset processing>
    if USE_ONLY_EVAL and os.path.isfile(f'{COLLECTION_PATH}{DATASET}_eval.json'):
        print(f"{DATASET}_eval.json found, loading it...")
        docs = ('eval', json.loads(open(f'{COLLECTION_PATH}{DATASET}_eval.json', encoding='ISO-8859-1').read()))
    else:
        print(f"Loading full dataset...")
        if not OVERRIDE_SAVED_JSON and os.path.isfile(f'{COLLECTION_PATH}{DATASET}.json'):
            print(f"{DATASET}.json found, loading it...")
            docs = json.loads(open(f'{COLLECTION_PATH}{DATASET}.json', encoding='ISO-8859-1').read())
        else:
            print(f"{DATASET}.json not found, parsing full dataset...")
            docs = parse_dataset()
        docs = ('full', docs)

        if USE_ONLY_EVAL:
            docs = ('eval', get_subset(docs[1], doc_index))
            print(f"Saving eval set to {DATASET}_eval.json...")
            with open(f'{COLLECTION_PATH}{DATASET}_eval.json', 'w', encoding='ISO-8859-1') as f:
                f.write(json.dumps(docs[1], indent=4))
    # </Dataset processing>

    evaluation(topics, docs, analyzer=stem_analyzer, scoring=NamedBM25F(K1=2, B=1))

    # tune_bm25("BM25tune_results_lemma.json", I, topic_index)
    return 0


def evaluation(Q, D, analyzer=None, scoring=None):
    global doc_index, doc_index_n, topic_index, topic_index_n
    # <Get fold>
    if FOLDS:
        doc_ids_list = list(D[1])
        kf = KFold(n_splits=FOLDS, random_state=RANDOM_STATE, shuffle=True)
        sampled_doc_ids = [doc_ids_list[i] for i in next(kf.split(doc_ids_list))[1]]
        print(len(sampled_doc_ids))
        doc_index = get_subset(doc_index, sampled_doc_ids)
        doc_index_n = get_subset(doc_index_n, sampled_doc_ids)

        topic_index = invert_index(doc_index)
        topic_index_n = invert_index(doc_index_n)
        D = (f"{D[0]}_{FOLDS}", get_subset(D[1], sampled_doc_ids))
    # </get fold>

    I, indexing_time, indexing_space = indexing(D, analyzer=analyzer, scoring=scoring)
    print(f'Indexing time: {indexing_time:10.3f}s, Indexing space: {indexing_space / (1024 ** 2):10.3f}mb')
    for q in tqdm(Q, desc=f'{f"BOOLEAN RETRIEVAL":20}'):
        # print("Topic:", q)
        # print('Topic:', topics[q], sep='\n')
        # print('Topic Keywords:', *extract_topic_query(q, I, k=5, metric='tfidf'), sep='\n')
        # doc_ids = boolean_query(q, I, k=5, metric='tfidf')
        # print("Relevant documents:", [I.test[doc_id] for doc_id in doc_ids])
        pass
    # old_ranking(I, topic_index, topics)

    # <Get Ranking results>
    results_file = f"{I.whoosh_dir.replace('whoosh', 'ranking_results')}_{I.scoring}.json"
    if not os.path.exists("ranking_results"):
        os.mkdir("ranking_results")
    if os.path.isfile(results_file):
        print(f"Ranking results already exist, loading from file (\"{results_file}\")...")
        precision_results = jsonpickle.decode(open(results_file, encoding='ISO-8859-1').read())
    else:
        print(f"Ranking results don't exist, ranking with model...")
        precision_results = rank_topics(I, topic_index, topic_index_n)
        with open(results_file, 'w', encoding='ISO-8859-1') as f:
            f.write(jsonpickle.encode(precision_results, indent=4))
    # </Get Ranking results>

    print_general_stats(precision_results, topic_index)

def rank_topics(I, topic_index, topic_index_n, scoring=None, leave=True):
    if scoring:
        I.scoring = scoring
    precision_results = {q_id: {'related_documents': set(doc_ids)} for q_id, doc_ids in topic_index.items()}
    for q in tqdm(topic_index, desc=f'{f"RANKING":20}', leave=leave):
        retrieved_doc_ids, retrieved_scores = zip(*ranking(q, 500, I))
        precision_result = {
            'total_result': len(retrieved_doc_ids),
            'visited_documents': retrieved_doc_ids,
            'visited_documents_orders': {doc_id: rank + 1 for rank, doc_id in enumerate(retrieved_doc_ids)},
            'assessed_documents': {doc_id: (rank + 1, int(doc_id in topic_index[q])) for rank, doc_id in enumerate(retrieved_doc_ids) if
                                   doc_id in topic_index.get(q, []) or doc_id in topic_index_n.get(q, [])}
        }
        precision_results[q].update(precision_result)

    return precision_results


def tune_bm25(BM25tune_file, I, topic_index):
    print(f"Tuning BM25 with {I.analyzer}...")
    if os.path.isfile(f'{COLLECTION_PATH}{BM25tune_file}'):
        results = pd.read_json(f'{COLLECTION_PATH}{BM25tune_file}', orient='split')
    else:
        results = None
    results_update = defaultdict(list)
    with tqdm(K1_TEST_VALS, desc=f'{f"TESTING k1={K1_TEST_VALS[0]:.2f}":20}') as k1_tqdm:
        for k1 in k1_tqdm:
            k1_tqdm.set_description(desc=f'{f"TESTING k1={k1:.2f}":20}')
            with tqdm(B_TEST_VALS, desc=f'{f"TESTING b={B_TEST_VALS[0]:.2f}":20}', leave=False) as b_tqdm:
                for b in b_tqdm:
                    b_tqdm.set_description(desc=f'{f"TESTING b={b:.2f}":20}')
                    if results is not None and not results.loc[(results['k1'] == k1) & (results['b'] == b)].empty:
                        continue

                    I.scoring = whoosh.scoring.BM25F(B=b, K1=k1)
                    precision_results = rank_topics(I, topic_index, leave=False)

                    metrics_scores = defaultdict(list)
                    for q_id, data in precision_results.items():
                        for metric, score in calc_precision_based_measures(data['visited_documents'], data['related_documents'], K_TESTS).items():
                            metrics_scores[metric].append(score)

                    for metric, scores in metrics_scores.items():
                        results_update[metric].append(np.mean(scores))
                    results_update['k1'].append(k1)
                    results_update['b'].append(b)
    update_data = pd.DataFrame(data=results_update)
    if results is None:
        results = update_data
    else:
        results.append(update_data, ignore_index=True)
    with open(f'{COLLECTION_PATH}{BM25tune_file}{I.analyzer}.json', 'w') as f:
        f.write(json.dumps(json.loads(results.to_json(orient='split')), indent=4))

    print(results)


def invert_index(index):
    inverted_index = defaultdict(list)
    for id, indexed_ids in index.items():
        for indexed_id in indexed_ids:
            inverted_index[indexed_id].append(id)
    return dict(inverted_index)


if __name__ == '__main__':
    main()
