import itertools
import networkx as nx
import pagerank as prank

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd


from metrics import *
from parsers import *

from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances

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


threshold = 0.4
SUBSET_SIZE = 1000
top_p = 10

def compute_similarity_matrix(d, similarity,th):
    tfidf_vectorizer = TfidfVectorizer()
    matrix = tfidf_vectorizer.fit_transform(' '.join(list(value.values())) for key,value in d.items())
    df = pd.DataFrame(matrix.todense(),
                      columns=tfidf_vectorizer.get_feature_names(),
                      index= d.keys())
    sim_matrix = np.array(similarity(df,df))
    for row in sim_matrix:
        row[row < th] = 0  # removes edges that dont meet the threshold
    print({'matrix shape:', sim_matrix.shape})
    return sim_matrix


def build_graph(D,sim,th):
    D = dict(itertools.islice(D.items(),SUBSET_SIZE)) if len(D) > 1000 else D # 1000 docs subset
    sim_matrix = compute_similarity_matrix(D,sim,th) # calculated the similarities into a matrix
    sim_graph = nx.from_numpy_matrix(sim_matrix) # transforms into graph
    print({'nr of edges': sim_graph.number_of_edges()})
    sim_graph.remove_edges_from(nx.selfloop_edges(sim_graph))

    #print(sim_graph.edges.data())

    # nx.draw(sim_graph)
    # plt.show()
    return sim_graph

def undirected_page_rank(q,D,p,sim,th):
    doc_ids = topic_index[0][q]
    sim_graph = build_graph(get_subset(D,doc_ids), sim, th)
    try:
        pr = prank.pagerank(sim_graph, max_iter=50, weight=None)
    except nx.PowerIterationFailedConvergence as e:
        print(e)
    return get_subset(pr,sorted(list(pr.keys()),key=lambda x:pr[x])[:p:-1])


def main():
    D = {}
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = setup()
    for doc in docs:
        D.update(docs[doc])
    for topic in topics:
        print(topic)
        pr_values = undirected_page_rank(topic, D, top_p, sim = cosine_similarity, th = threshold)
        print({topic: pr_values})

if __name__ == '__main__':
    main()