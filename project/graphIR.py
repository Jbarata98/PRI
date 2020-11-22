import itertools
import networkx as nx

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

from metrics import *
from parsers import *

from sklearn.metrics.pairwise import cosine_similarity

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


threshold = 0.01
SUBSET_SIZE = 1000

def compute_similarity_matrix(d):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    matrix = tfidf_vectorizer.fit_transform(value['p'] for key,value in d.items())
    df = pd.DataFrame(matrix.todense(),
                      columns=tfidf_vectorizer.get_feature_names(),
                      index= d.keys())
    return np.array(cosine_similarity(df, df))


def build_graph(d,th):
    subset = dict(itertools.islice(d.items(),SUBSET_SIZE))
    sim_matrix = compute_similarity_matrix(subset)
    G = nx.from_numpy_matrix(sim_matrix)
    G.remove_edges_from((e for e, w in nx.get_edge_attributes(G, 'weight').items() if w < th))

    nx.draw(G)
    plt.show()



def main():
    D = {}
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = setup()
    for doc in docs:
        D.update(docs[doc])
    build_graph(D,th = threshold)

if __name__ == '__main__':
    main()