import random
import main as p1
from typing import List

import scipy as sy

import jsonpickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy.sparse import hstack

from BM25Vectorizer import BM25Vectorizer
from metrics import *
from parsers import *

OVERRIDE_SAVED_JSON = False

topics = None
docs = None
topic_index = None
doc_index = None

confusion_matrix = [
    ["TP", 'FN'],
    ["FP", 'TN']
]


class NamedClassifier():
    def __init__(self, classifier, name, file_term=None):
        self.classifier = classifier
        self.name = name
        self.file_term = file_term if file_term else name.replace(' ', '-').lower()

    def __call__(self, *args, **aargs):
        return self.classifier(*args, **aargs)

    def __repr__(self):
        return repr(self.classifier)

    def __str__(self):
        return self.name

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    @property
    def classes(self):
        return self.classifier.classes_


class NamedVectorizer():
    def __init__(self, vectorizer, name, file_term=None):
        self.vectorizer = vectorizer
        self.name = name
        self.file_term = file_term if file_term else name.replace(' ', '-').lower()

    def __call__(self, *args, **aargs):
        return self.vectorizer(*args, **aargs)

    def __repr__(self):
        return repr(self.vectorizer)

    def __str__(self):
        return self.name

    def fit(self, X, y=None):
        self.vectorizer.fit(X, y)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X)


class StaticVectorizer(NamedVectorizer):
    def __init__(self, vectorizer, name, file_term=None):
        super().__init__(vectorizer, name, file_term)
        self.name = 'Static ' + self.name
        self.file_term = 'static-' + self.file_term

    def fit_transform(self, X):
        return self.vectorizer.transform(X)


class EnsembleVectorizer(NamedVectorizer):
    def __init__(self, *vectorizers):
        super().__init__(None, "Ensemble of " + ' + '.join(v.name for v in vectorizers), "ensemble_" + '+'.join(v.file_term for v in vectorizers))
        self.vectorizers = vectorizers

    def fit(self, X, y=None):
        for vectorizer in self.vectorizer:
            vectorizer.fit(X, y)
        return self

    def transform(self, X):
        return hstack([vectorizer.transform(X) for vectorizer in self.vectorizers])

    def fit_transform(self, X):
        x = hstack([vectorizer.fit_transform(X) for vectorizer in self.vectorizers])
        return x


class SparseVectorClassifier:
    def __init__(self, D, R, vectorizer=None, classifier=None):
        self.tfidf_index = vectorizer if vectorizer else TfidfVectorizer()
        self.tfidf_test_matrix = self.tfidf_index.fit_transform(self.raw_text_from_dict(D))
        self.classifier = classifier
        self.fit(self.tfidf_test_matrix, R)

    @staticmethod
    def raw_text_from_dict(doc_dict):
        return [' '.join(list(doc.values())) for doc in doc_dict.values()]  # Joins all docs in a single list of raw_doc strings

    @property
    def idf(self):
        return self.tfidf_index.idf_

    @property
    def classes(self):
        return self.classifier.classes

    @property
    def doc_ids(self):
        return [doc_id for doc_id in self.D.keys()]

    def get_term_idf(self, term):
        return 0 if term not in self.vocabulary else self.idf[self.vocabulary[term]]

    def get_matrix_data(self):
        return {'len': len(self.idf), 'vocabulary': self.vocabulary, 'idf': self.idf}

    def transform(self, raw_documents):
        return self.tfidf_index.transform(raw_documents)

    def build_analyzer(self):
        return self.tfidf_index.build_analyzer()

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.classifier.predict_proba(self.transform(X))


mlp_classifier = NamedClassifier(MLPClassifier(random_state=1, max_iter=1000), "MLP", "mlp")
mnb_classifier = NamedClassifier(MultinomialNB(), 'Multinomial Naïve Bayes', "mnb")
knn_classifier = NamedClassifier(KNeighborsClassifier(n_neighbors=3), 'KNN', "knn")

tfidf_vectorizer = NamedVectorizer(TfidfVectorizer(), 'TF-IDF')
tf_vectorizer = NamedVectorizer(CountVectorizer(), 'TF')
bm25_vectorizer = NamedVectorizer(BM25Vectorizer(), 'BM25')
static_tfidf_vectorizer = StaticVectorizer(TfidfVectorizer(), 'TF-IDF')
tf_and_idf_vectorizer = EnsembleVectorizer(tfidf_vectorizer, tf_vectorizer)
tf_and_idf_and_bm25_vectorizer = EnsembleVectorizer(tfidf_vectorizer, tf_vectorizer, bm25_vectorizer)


def raw_text_from_dict(doc_dict):
    return [' '.join(list(doc.values())) for doc in doc_dict.values()]  # Joins all docs in a single list of raw_doc strings


def get_subset(adict, subset):
    return {key: adict[key] for key in subset if key in adict}


def setup():
    global topics, topic_index, doc_index, topic_index_n, doc_index_n

    # EXTRACTION
    extract_dataset()

    # <Build Q>
    topics = parse_topics(f"{COLLECTION_PATH}{TOPICS}")
    train_indexes = parse_qrels(f"{COLLECTION_PATH}qrels.train.txt")
    for train_index, test_index in zip(train_indexes, parse_qrels(f"{COLLECTION_PATH}qrels.test.txt")):
        for key in test_index:
            if key in train_index:
                train_index[key] += test_index[key]
            else:
                train_index[key] = test_index[key]

    topic_index, doc_index, topic_index_n, doc_index_n = train_indexes
    topic_index = {'p': topic_index, 'n': topic_index_n}
    doc_index = {'p': doc_index, 'n': doc_index_n}
    # </Build Q>

    # <Dataset processing>
    docs = {}

    for split in ('train', 'test'):
        print(f'\n\nPARSING {split.upper()} SET...')
        if not OVERRIDE_SAVED_JSON and os.path.isfile(f'{COLLECTION_PATH}{DATASET}_eval_{split}.json'):
            print(f"{DATASET}_eval_{split}.json found, loading them...")
            docs[split] = json.loads(open(f'{COLLECTION_PATH}{DATASET}_eval_{split}.json', encoding='ISO-8859-1').read())
        else:
            if os.path.isfile(f'{COLLECTION_PATH}{DATASET}_{split}.json'):
                print(f"{DATASET}_{split}.json found, loading it...")
                docs[split] = dict(json.loads(open(f'{COLLECTION_PATH}{DATASET}_{split}.json', encoding='ISO-8859-1').read()))
            else:
                print(f"{DATASET}_{split}.json not found, parsing full dataset...")
                docs[split] = (parse_dataset(split), doc_index)
            docs[split] = get_subset(docs[split], list(doc_index['p'].keys()) + list(doc_index['n'].keys()))
            print(f"Saving eval set to {DATASET}_eval_{split}.json...")
            with open(f'{COLLECTION_PATH}{DATASET}_eval_{split}.json', 'w', encoding='ISO-8859-1') as f:
                f.write(json.dumps(docs[split], indent=4))
        # </Dataset processing>

    return docs, topics, topic_index, doc_index


def training(q, Dtrain, Rtrain, classifier=None, vectorizer=None, **args):
    q_judged_doc_ids = Rtrain['p'].get(q, []) + Rtrain['n'].get(q, [])
    q_judged_docs = get_subset(Dtrain, q_judged_doc_ids)
    return SparseVectorClassifier(q_judged_docs, [int(doc_id in Rtrain['p'].get(q, [])) for doc_id in q_judged_docs], vectorizer=vectorizer, classifier=classifier, **args)


def classify(d: dict, q: str, M: SparseVectorClassifier, **args):
    return M.predict_proba([' '.join(d.values())])[0][np.where(M.classes == 1)][0]


def entropy(p):
    # return 0 if p in (0, 1) else -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)
    return min(p, 1 - p)


def evaluate(Qtest, Dtest, Rtest, classifiers: List[NamedClassifier] = (mlp_classifier,), vectorizers: List[NamedVectorizer] = (tfidf_vectorizer,), **args):
    total_results, models_ranking_results = {}, {}
    for vectorizer in vectorizers:
        for classifier in classifiers:

            print(f"Classifying with classifier: {classifier}, and vectorizer: {vectorizer}:")
            classification_results = get_classification_results(Dtest, Qtest, Rtest, classifier, vectorizer)
            title = f'{classifier} & {vectorizer}'
            models_ranking_results[f'{classifier} & {vectorizer}'] = classification_results

            results = defaultdict(dict)
            for q, result in classification_results.items():
                results[q]['tp'] = len(result['predicted_related'].intersection(result['related_documents']))
                results[q]['fn'] = len(result['predicted_unrelated'].intersection(result['related_documents']))
                results[q]['fp'] = len(result['predicted_related'].intersection(result['unrelated_documents']))
                results[q]['tn'] = len(result['predicted_unrelated'].intersection(result['unrelated_documents']))

                correct = results[q]['tp'] + results[q]['tn']
                results[q]['accuracy'] = (correct / (correct + results[q]['fp'] + results[q]['fn']))
                results[q]['1-entropy'] = 1 - entropy(len(result['related_documents']) / len(result['assessed_documents']))
                results[q]['sensitivity'] = (results[q]['tp'] / max(results[q]['tp'] + results[q]['fn'], 1))
                results[q]['specificity'] = (results[q]['tn'] / max(results[q]['tn'] + results[q]['fp'], 1))

            metrics = {'1-entropy': [], 'accuracy': [], 'sensitivity': [], 'specificity': []}
            ids_sorted_by_entropy = sorted(classification_results, key=lambda a: results[a]['1-entropy'])
            for metric in metrics:
                metrics[metric] = [results[q_id][metric] for q_id in ids_sorted_by_entropy]
            total_results[title] = metrics['accuracy']
            total_results['1-entropy'] = metrics['1-entropy']
            plt.figure(figsize=(15, 5))
            multiple_line_chart(plt.gca(), ids_sorted_by_entropy, metrics, f"Classification performance statistics for {classifier} with {vectorizer}", "Topics", "Percentage", show_points=True,
                                ypercentage=True)
            plt.show()
            plt.figure(figsize=(15, 5))
            metrics_per_sorted_topic(classification_results, title)

    plt.figure(figsize=(15, 5))
    multiple_line_chart(plt.gca(), ids_sorted_by_entropy, total_results, f"Classification accuracy statistics for multiple approaches", "Topics", "Accuracy", show_points=True,
                        ypercentage=True)
    plt.show()

    p1.topics = topics
    models_ranking_results.update(p1.evaluation(Qtest, (p1.invert_index(Rtest['p']), p1.invert_index(Rtest['n'])), ('test', Dtest), (p1.stem_analyzer,), (p1.NamedBM25F(K1=2, B=1),)))

    plt.figure(figsize=(15, 5))
    plot_iap_for_models(models_ranking_results)

    return dict(classification_results)


def get_classification_results(Dtest, Qtest, Rtest, classifier, vectorizer):
    # <Get Retrieval results>
    classification_results_file = f"classification_results/eval_{classifier.file_term}_{vectorizer.file_term}"
    if not os.path.exists("classification_results"):
        os.mkdir("classification_results")
    if os.path.isfile(classification_results_file):
        print(f"Retrieval results already exist, loading from file (\"{classification_results_file}\")...")
        classification_results = jsonpickle.decode(open(classification_results_file, encoding='ISO-8859-1').read())
    else:
        print(f"Retrieval results don't exist, retrieving with model...")
        classification_results = classify_topics(Dtest, Qtest, Rtest, classifier=classifier, vectorizer=vectorizer)
        with open(classification_results_file, 'w', encoding='ISO-8859-1') as f:
            f.write(jsonpickle.encode(classification_results, indent=4))
    # </Get Retrieval results>
    return classification_results


def classify_topics(Dtest, Qtest, Rtest, classifier: NamedClassifier = None, vectorizer=None):
    raw_results = {}
    ranking_results = {q_id: {'related_documents': set(doc_ids)} for q_id, doc_ids in Rtest['p'].items()}
    with tqdm(Qtest, desc=f'{f"CLASSIFYING {list(Qtest)[0]}":20}', leave=False) as q_tqdm:
        for q in q_tqdm:
            q_tqdm.set_description(desc=f'{f"CLASSIFYING {q}":20}')

            model = training(q, docs['train'], Rtest, classifier=classifier, vectorizer=vectorizer)
            raw_results = {'p': {d_id: classify(doc, q, model) for d_id, doc in get_subset(Dtest, Rtest['p'].get(q, [])).items()},
                           'n': {d_id: classify(doc, q, model) for d_id, doc in get_subset(Dtest, Rtest['n'].get(q, [])).items()}}
            retrieved_doc_ids = dict(sorted({**raw_results['p'], **raw_results['n']}.items(), key=lambda x: x[1], reverse=True))
            ranking_result = {
                'unrelated_documents': set(Rtest['n'].get(q, [])),
                'total_result': len(retrieved_doc_ids),
                'visited_documents': list(retrieved_doc_ids),
                'visited_documents_orders': {doc_id: rank + 1 for rank, doc_id in enumerate(retrieved_doc_ids)},
                'assessed_documents': {doc_id: (rank + 1, int(doc_id in Rtest['p'].get(q, []))) for rank, doc_id in enumerate(retrieved_doc_ids) if
                                       doc_id in Rtest['p'].get(q, []) or doc_id in Rtest['n'].get(q, [])},
                'document_probabilities': retrieved_doc_ids,
                'predicted_related': set([doc_id for doc_id, prob in retrieved_doc_ids.items() if round(prob)]),
                'predicted_unrelated': set([doc_id for doc_id, prob in retrieved_doc_ids.items() if not round(prob)]),
            }
            ranking_results[q].update(ranking_result)
    return ranking_results


def main():
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = setup()
    print('ola joao Barata :)\n' + ''.join([random.choice(("\u5350", "\u534d")) for _ in range(1000)]), "\n    __\n|__|__\n __|  |")
    # static_tfidf_vectorizer.fit([' '.join(list(doc.values())) for doc in docs['train'].values()] )
    evaluate(topics, docs['test'], topic_index, classifiers=(mlp_classifier, knn_classifier), vectorizers=(tf_vectorizer, tfidf_vectorizer))
    print("ADEUS BARATA")


if __name__ == '__main__':
    main()
