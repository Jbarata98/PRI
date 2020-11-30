import abc
import random
from copy import deepcopy
from enum import Enum

from sklearn.model_selection import GridSearchCV

import main as p1
from typing import List

import scipy as sy

import jsonpickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy.sparse import hstack

from BM25Vectorizer import *
from metrics import *
from parsers import *

OVERRIDE_SAVED_JSON = False
TESTED_N_NEIGHBOURS = (1, 3, 5, 7)
TESTED_KNN_DISTANCES = ('euclidean', 'manhattan')
TESTED_LAYER_COMPS = (
    (25,),
    (50,),
    (100,),
    (25, 100),
    (100, 25),
    (25, 50, 100),
    (100, 50, 25),
)

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
        super().__init__(None, ' + '.join(v.name for v in vectorizers), "ensemble_" + '+'.join(v.file_term for v in vectorizers))
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


# TESTED CLASSIFIERS
mlp_classifier = NamedClassifier(MLPClassifier(random_state=1, max_iter=1000), "MLP", "mlp")  # unused but working
mnb_classifier = NamedClassifier(MultinomialNB(), 'Multinomial Naïve Bayes', "mnb")
knn_classifier = NamedClassifier(KNeighborsClassifier(n_neighbors=3), 'KNN', "knn")

# TESTED TUNING
tuned_knn_classifier = NamedClassifier(GridSearchCV(KNeighborsClassifier(), {'n_neighbors': TESTED_N_NEIGHBOURS, 'metric': TESTED_KNN_DISTANCES}, verbose=0, cv=3), 'Tuned KNN')
tuned_mlp_classifier = NamedClassifier(GridSearchCV(MLPClassifier(max_iter=500), {'hidden_layer_sizes': TESTED_LAYER_COMPS}, verbose=0, cv=3), 'Tuned MLP')

# TESTED VECTORIZERS
tfidf_vectorizer = NamedVectorizer(TfidfVectorizer(), 'TF-IDF')
tf_vectorizer = NamedVectorizer(CountVectorizer(), 'TF')
bm25_vectorizer = NamedVectorizer(BM25Vectorizer(), 'BM25')
simple_vectorizers = (tfidf_vectorizer, bm25_vectorizer, tf_vectorizer)

# SPECIAL VECTORIZERS
bm25_scorer = NamedVectorizer(BM25Scorer(), 'BM25 scorer')  # BM25 train document scores
static_tfidf_vectorizer = StaticVectorizer(TfidfVectorizer(), 'static TF-IDF')  # A TFIDF vectorizer where the whole trainning set is indexed and is shared between topics

# EMSEMBLED VECTORIZERS (MULTIPLE IR MODELS)
tf_and_idf_vectorizer = EnsembleVectorizer(tfidf_vectorizer, tf_vectorizer)
tf_and_bm25_vectorizer = EnsembleVectorizer(bm25_vectorizer, tf_vectorizer)
tfidf_and_bm25_vectorizer = EnsembleVectorizer(bm25_vectorizer, tfidf_vectorizer)
tf_and_idf_and_bm25_vectorizer = EnsembleVectorizer(tfidf_vectorizer, tf_vectorizer, bm25_vectorizer)
emsembled_vectorizers = (tf_and_idf_vectorizer, tf_and_bm25_vectorizer, tfidf_and_bm25_vectorizer, tf_and_idf_and_bm25_vectorizer)


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
    values_ = M.predict_proba([' '.join(d.values())])[0]
    return values_[np.where(M.classes == 1)][0]


def entropy(p):
    # return 0 if p in (0, 1) else -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)
    return 1 - p


def evaluate(Qtest, Dtest, Rtest, models=((tfidf_vectorizer, mlp_classifier),), ranking_results=None, retrieval_results=None,
             **args):
    total_results, models_classification_results, models_ranking_results = {}, {}, {}
    for vectorizer, classifier in models:
        reranking_results = None

        print(f"\nClassifying with classifier: {classifier}, and vectorizer: {vectorizer}:")
        if ranking_results:
            reranking_results = deepcopy(ranking_results)
        classification_results = get_classification_results(Dtest, Qtest, Rtest, classifier, vectorizer, reranking_results)
        title = f'{classifier} & {vectorizer}'
        models_classification_results[title] = classification_results

        ids_sorted_by_entropy, metrics = plot_classification_metrics(classification_results, title)

        total_results[title] = metrics['accuracy']
        total_results['Negative Bias'] = metrics['negative bias']

        if reranking_results:
            plt.figure(figsize=(15, 5))
            metrics_per_sorted_topic(reranking_results, title)

            plt.figure(figsize=(7, 5))
            print_general_stats(reranking_results, title)
            models_ranking_results[title] = reranking_results

    if retrieval_results:
        title = f'Retrieval Baseline'
        ids_sorted_by_entropy, metrics = plot_classification_metrics(retrieval_results, title)
        total_results[title] = metrics['accuracy']

    plt.figure(figsize=(15, 5))
    multiple_line_chart(plt.gca(), ids_sorted_by_entropy, total_results, f"Classification accuracy statistics for multiple approaches", "Topics", "Accuracy", show_points=True,
                        ypercentage=True)
    plt.rc('grid', color='grey', linewidth=1, alpha=0.3)
    plt.grid(axis='x')
    plt.show()

    if models_ranking_results:
        title = f'Ranking Baseline'
        models_ranking_results[title] = ranking_results

        plt.figure(figsize=(15, 5))
        metrics_per_sorted_topic(ranking_results, title)

        plt.figure(figsize=(7, 5))
        print_general_stats(ranking_results, title)

        plt.figure(figsize=(15, 5))
        results_metric_per_sorted_topic(models_ranking_results, 'map', title)

        plt.figure(figsize=(15, 5))
        results_metric_per_sorted_topic(models_ranking_results, 'precision@10', title)

        plt.figure(figsize=(7, 5))
        plot_iap_for_models(models_ranking_results)

    return dict(classification_results)


def plot_classification_metrics(classification_results, title):
    results = defaultdict(dict)
    for q, result in classification_results.items():
        results[q]['tp'] = len(result['predicted_related'].intersection(result['related_documents']))
        results[q]['fn'] = len(result['predicted_unrelated'].intersection(result['related_documents']))
        results[q]['fp'] = len(result['predicted_related'].intersection(result['unrelated_documents']))
        results[q]['tn'] = len(result['predicted_unrelated'].intersection(result['unrelated_documents']))

        correct = results[q]['tp'] + results[q]['tn']
        results[q]['accuracy'] = (correct / (correct + results[q]['fp'] + results[q]['fn']))
        results[q]['negative bias'] = entropy(len(result['related_documents']) / len(result['assessed_documents']))
        results[q]['sensitivity'] = (results[q]['tp'] / max(results[q]['tp'] + results[q]['fn'], 1))
        results[q]['specificity'] = (results[q]['tn'] / max(results[q]['tn'] + results[q]['fp'], 1))
    metrics = {'negative bias': [], 'accuracy': [], 'sensitivity': [], 'specificity': []}
    ids_sorted_by_entropy = sorted(classification_results, key=lambda a: results[a]['negative bias'])
    for metric in metrics:
        metrics[metric] = [results[q_id][metric] for q_id in ids_sorted_by_entropy]
    plt.figure(figsize=(15, 5))
    multiple_line_chart(plt.gca(), ids_sorted_by_entropy, metrics, f"Classification performance statistics for {title}", "Topics", "Percentage", show_points=True,
                        ypercentage=True)
    plt.rc('grid', color='grey', linewidth=1, alpha=0.3)
    plt.grid(axis='x')
    plt.show()
    return ids_sorted_by_entropy, metrics


def get_classification_results(Dtest, Qtest, Rtest, classifier, vectorizer, pre_retrieval=None, skip_classification=False):
    classification_results, classification_exists = None, False

    classification_results_file = f"classification_results/eval_{classifier.file_term}_{vectorizer.file_term}.json"
    reranking_results_file = f"reranking_results/eval_{classifier.file_term}_{vectorizer.file_term}.json"
    if not os.path.exists("classification_results"):
        os.mkdir("classification_results")
    if not os.path.exists("reranking_results"):
        os.mkdir("reranking_results")

    # Classification checkpoint exists
    if (not skip_classification) and os.path.isfile(classification_results_file):
        print(f"Classification results already exist, loading from file (\"{classification_results_file}\")...")
        classification_results = jsonpickle.decode(open(classification_results_file, encoding='ISO-8859-1').read())
        classification_exists = True
    elif not skip_classification and not pre_retrieval:
        print(f"Classification results don't exist, retrieving with model...")
        classification_results = classify_topics(Dtest, Qtest, Rtest, classifier=classifier, vectorizer=vectorizer, pre_retrieval=None, skip_classification=skip_classification)
        with open(reranking_results_file, 'w', encoding='ISO-8859-1') as f:
            f.write(jsonpickle.encode(pre_retrieval, indent=4))

    if not pre_retrieval:
        return classification_results

    # Ranking checkpoint exists
    if os.path.isfile(reranking_results_file):
        print(f"Reranking results already exist, loading from file (\"{reranking_results_file}\")...")
        pre_retrieval.update(jsonpickle.decode(open(reranking_results_file, encoding='ISO-8859-1').read()))
        if not (classification_exists or skip_classification):
            print(f"Classification results don't exist, retrieving with model...")
            classification_results = classify_topics(Dtest, Qtest, Rtest, classifier=classifier, vectorizer=vectorizer, pre_retrieval=None, skip_classification=skip_classification)
    else:
        print(f"Reranking results don't exist, retrieving with model...")
        classification_results = classify_topics(Dtest, Qtest, Rtest, classifier=classifier, vectorizer=vectorizer, pre_retrieval=pre_retrieval, skip_classification=skip_classification)
        print(f"Saving reranking to file (\"{reranking_results_file}\")...")
        with open(reranking_results_file, 'w', encoding='ISO-8859-1') as f:
            f.write(jsonpickle.encode(pre_retrieval, indent=4))

    # Save results
    if not classification_exists and classification_results:
        print(f"Saving classification to file (\"{classification_results_file}\")...")
        with open(classification_results_file, 'w', encoding='ISO-8859-1') as f:
            f.write(jsonpickle.encode(classification_results, indent=4))

    return classification_results


def classify_topics(Dtest, Qtest, Rtest, classifier: NamedClassifier = None, vectorizer=None, k=DEFAULT_K, pre_retrieval=None, skip_classification=False):
    classification_results = {q_id: {'related_documents': set(doc_ids)} for q_id, doc_ids in Rtest['p'].items()}
    with tqdm(Qtest, desc=f'{f"CLASSIFYING {list(Qtest)[0]}":20}', leave=True, dynamic_ncols=True) as q_tqdm:
        for q in q_tqdm:
            q_tqdm.set_description(desc=f'{f"CLASSIFYING {q}":20}')

            model = training(q, docs['train'], Rtest, classifier=classifier, vectorizer=vectorizer)

            # CLASSIFICATION
            if not skip_classification:
                raw_results = {'p': {d_id: classify(doc, q, model) for d_id, doc in get_subset(Dtest, Rtest['p'].get(q, [])).items()},
                               'n': {d_id: classify(doc, q, model) for d_id, doc in get_subset(Dtest, Rtest['n'].get(q, [])).items()}}
                retrieved_doc_ids = dict(sorted({**raw_results['n'], **raw_results['p']}.items(), key=lambda x: x[1], reverse=True))
                classification_result = {
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
                classification_results[q].update(classification_result)

            # RANKING
            if pre_retrieval:
                retrieved_docs_ids = dict(sorted({d_id: classify(doc, q, model) for d_id, doc in get_subset(Dtest, pre_retrieval[q]['visited_documents']).items()}
                                                 .items(), key=lambda x: x[1], reverse=True))

                ranking_result = {
                    'visited_documents': list(retrieved_docs_ids),
                    'visited_documents_orders': {doc_id: rank + 1 for rank, doc_id in enumerate(retrieved_docs_ids)},
                    'document_probabilities': retrieved_docs_ids
                }
                pre_retrieval[q].update(ranking_result)

    return classification_results


get_all_combinations = lambda xs, ys: [(x, y) for x in xs for y in ys]


class Experiment(Enum):
    mlp = 0
    tuned_mlp = 2
    knn = 3
    tuned_knn = 4
    compare_simple = 5
    emsembles_mlp = 6
    emsembles_knn = 7
    ablation_knn_neighbours = 8
    ablation_knn_distances = 9
    ablation_mlp_1_layer_comps = 10
    ablation_mlp_2_layer_comps = 11
    ablation_mlp_3_layer_comps = 12


def main(experiment=Experiment.compare_simple):
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = setup()
    p1.topics = topics
    try:
        p1_results = p1.evaluation(topics, (doc_index['p'], doc_index['n']), ('test', docs['test']), (p1.stem_analyzer,), (p1.NamedBM25F(K1=2, B=1),), 'tfidf', skip_indexing=True)
    except Exception as e:
        p1_results = p1.evaluation(topics, (doc_index['p'], doc_index['n']), ('test', docs['test']), (p1.stem_analyzer,), (p1.NamedBM25F(K1=2, B=1),), 'tfidf', skip_indexing=False)

    p1_ranking = list(p1_results[0].values())[0]
    p1_retrieval = list(p1_results[1].values())[0]

    experimental_evaluate = lambda models: evaluate(topics, docs['test'], topic_index, models=models, ranking_results=p1_ranking,
                                                    retrieval_results=p1_retrieval)
    if experiment == experiment:
        experimental_evaluate(get_all_combinations(simple_vectorizers, [mlp_classifier]))

    elif experiment == Experiment.tuned_mlp:
        experimental_evaluate(get_all_combinations(simple_vectorizers, [tuned_mlp_classifier]))

    elif experiment == Experiment.knn:
        experimental_evaluate(get_all_combinations(simple_vectorizers, [knn_classifier]))

    elif experiment == Experiment.tuned_knn:
        experimental_evaluate(get_all_combinations(simple_vectorizers, [tuned_knn_classifier]))

    elif experiment == Experiment.compare_simple:
        experimental_evaluate([(tfidf_vectorizer, tuned_mlp_classifier), (tfidf_vectorizer, tuned_knn_classifier)])

    elif experiment == Experiment.emsembles_knn:
        experimental_evaluate(get_all_combinations(emsembled_vectorizers + (tfidf_vectorizer,), [tuned_knn_classifier]))

    elif experiment == Experiment.emsembles_mlp:
        experimental_evaluate(get_all_combinations(emsembled_vectorizers + (tfidf_vectorizer,), [tuned_mlp_classifier]))

    elif experiment == Experiment.ablation_knn_neighbours:
        knn_n_neighbour_variant_classifiers = [NamedClassifier(GridSearchCV(KNeighborsClassifier(n_neighbors=k), {'metric': TESTED_KNN_DISTANCES}, verbose=0, cv=3), f'Tuned {k}NN') for k in
                                               TESTED_N_NEIGHBOURS]
        experimental_evaluate(get_all_combinations((tfidf_vectorizer,), knn_n_neighbour_variant_classifiers))

    elif experiment == Experiment.ablation_knn_distances:
        knn_distance_variant_classifiers = [NamedClassifier(GridSearchCV(KNeighborsClassifier(metric=distance), {'n_neighbors': TESTED_N_NEIGHBOURS}, verbose=0, cv=3), f'Tuned KNN {distance}') for
                                            distance in TESTED_KNN_DISTANCES]
        experimental_evaluate(get_all_combinations((tfidf_vectorizer,), knn_distance_variant_classifiers))

    elif experiment == Experiment.ablation_mlp_1_layer_comps:
        mlp_1_layer_comp_variant_classifiers = [NamedClassifier(MLPClassifier(hidden_layer_sizes=layer_comp, random_state=1, max_iter=1000), f'MLP {layer_comp}', f'mlp_{layer_comp[0]}') for
                                                layer_comp in TESTED_LAYER_COMPS if len(layer_comp) == 1]
        experimental_evaluate(get_all_combinations((tfidf_vectorizer,), mlp_1_layer_comp_variant_classifiers))

    elif experiment == Experiment.ablation_mlp_2_layer_comps:
        mlp_2_layer_comp_variant_classifiers = [
            NamedClassifier(MLPClassifier(hidden_layer_sizes=layer_comp, random_state=1, max_iter=1000), f'MLP {layer_comp}', f'mlp_{layer_comp[0]}_{layer_comp[1]}') for
            layer_comp in TESTED_LAYER_COMPS if len(layer_comp) == 2]
        experimental_evaluate(get_all_combinations((tfidf_vectorizer,), mlp_2_layer_comp_variant_classifiers))

    elif experiment == Experiment.ablation_mlp_3_layer_comps:
        mlp_3_layer_comp_variant_classifiers = [
            NamedClassifier(MLPClassifier(hidden_layer_sizes=layer_comp, random_state=1, max_iter=1000), f'MLP {layer_comp}', f'mlp_{layer_comp[0]}_{layer_comp[1]}_{layer_comp[2]}') for
            layer_comp in TESTED_LAYER_COMPS if len(layer_comp) == 3]
        experimental_evaluate(get_all_combinations((tfidf_vectorizer,), mlp_3_layer_comp_variant_classifiers))

    else:
        print("Insert a valid experiment")

    # experimental_evaluate(get_all_combinations([bm25_vectorizer, tfidf_vectorizer], [mlp_classifier, knn_classifier]), ranking_results=p1_ranking,retrieval_results=p1_retrieval)


if __name__ == '__main__':
    main()
