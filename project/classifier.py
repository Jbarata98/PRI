import random

from parsers import *
from metrics import *

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

OVERRIDE_SAVED_JSON = False


topics = None
docs = None
topic_index = None
doc_index = None

confusion_matrix = [
    ["TP", 'FN'],
    ["FP", 'TN']
]


class SparseVectorClassifier:
    def __init__(self, D, R, vectorizer=None, classifier=None):
        self.tfidf_index = vectorizer if vectorizer else TfidfVectorizer()
        self.tfidf_test_matrix = self.tfidf_index.fit_transform(tqdm(self.raw_text_from_dict(D), desc=f'{"INDEXING TFIDF":20}'))
        self.classifier = MultinomialNB()
        self.fit(self.tfidf_test_matrix, R)

    @staticmethod
    def raw_text_from_dict(doc_dict):
        return [' '.join(list(doc.values())) for doc in doc_dict.values()]  # Joins all docs in a single list of raw_doc strings

    @property
    def idf(self):
        return self.tfidf_index.idf_

    @property
    def classes(self):
        return self.classifier.classes_

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

    def transform(self, raw_documents):
        return self.tfidf_index.transform(raw_documents)

    def build_analyzer(self):
        return self.tfidf_index.build_analyzer()

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.classifier.predict_proba(self.transform(X))


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
    topic_index = topic_index, topic_index_n
    doc_index = doc_index, doc_index_n
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
            docs[split] = get_subset(docs[split], list(doc_index[0].keys()) + list(doc_index[1].keys()))
            print(f"Saving eval set to {DATASET}_eval_{split}.json...")
            with open(f'{COLLECTION_PATH}{DATASET}_eval_{split}.json', 'w', encoding='ISO-8859-1') as f:
                f.write(json.dumps(docs[split], indent=4))
        # </Dataset processing>

    return docs, topics, topic_index, doc_index


def training(q, Dtrain, Rtrain, **args):
    q_judged_doc_ids = Rtrain[0].get(q, []) + Rtrain[1].get(q, [])
    q_judged_docs = get_subset(Dtrain, q_judged_doc_ids)
    return SparseVectorClassifier(q_judged_docs, [int(doc_id in Rtrain[0][q]) for doc_id in q_judged_docs], vectorizer=None, **args)


def classify(d: dict, q: str, M: SparseVectorClassifier, **args):
    return M.predict_proba([' '.join(d.values())])[0][np.where(M.classes == 1)][0]

def entropy(p):
    return 0 if p in (0,1) else -(p*math.log(p,2))+((1-p)*math.log(1-p,2))

def evaluate(Qtest, Dtest, Rtest, **args):
    results, raw_results = defaultdict(dict), {}
    metrics = defaultdict(list)
    for q in Qtest:
        model = training(q, docs['train'], Rtest)
        sub_test = Rtest[0].get(q, [])
        raw_results[q] = {d_id: classify(doc, q, model) for d_id, doc in get_subset(Dtest, sub_test).items()}
        labels = [round(res, 0) for res in raw_results[q].values()]
        results[q]['tp'] = labels.count(1)
        results[q]['fn'] = labels.count(0)

        sub_test = Rtest[1].get(q, [])
        raw_results[q] = {d_id: classify(doc, q, model) for d_id, doc in get_subset(Dtest, sub_test).items()}
        labels = [round(res, 0) for res in raw_results[q].values()]
        results[q]['fp'] = labels.count(1)
        results[q]['tn'] = labels.count(0)

        correct = results[q]['tp'] + results[q]['tn']
        metrics['entropy'].append(entropy(len( Rtest[0].get(q, []))/( len(Rtest[0].get(q, []))+ len(Rtest[1].get(q, [])))))
        metrics['accuracy'].append(correct / (correct + results[q]['fp'] + results[q]['fn']))
        # metrics['sensitivity'].append((results[q]['tp'] / max(results[q]['tp'] + results[q]['fn'],1)))
        # metrics['specificity'].append((results[q]['tn'] / max(results[q]['tn'] + results[q]['fp'],1)))
        print(results[q])
    multiple_line_chart(plt.gca(),list(Qtest.keys()),metrics,"Classification performance statistics", "Topics", "%Accuracy",show_points= True, ypercentage= True)
    plt.show()

    return dict(results), raw_results





def main():
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = setup()
    print('ola joao Barata :)\n' + ''.join([random.choice(("\u5350", "\u534d")) for _ in range(1000)]))
    evaluate(topics, docs['test'], topic_index)
    print("ADEUS BARATA")


if __name__ == '__main__':
    main()
