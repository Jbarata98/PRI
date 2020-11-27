import numpy as np
from metrics import *
from parsers import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score
from scipy.cluster.hierarchy import dendrogram

COLLECTION_LEN = 807168
COLLECTION_PATH = 'collection/'
DATASET = 'rcv1'
TRAIN_DATE_SPLIT = '19960930'
AVAILABLE_DATA = ('headline', 'p', 'dateline', 'byline')
DATA_HEADER = ('newsitem', 'itemid')
TOPICS = "topics.txt"
TOPICS_CONTENT = ('num', 'title', 'desc', 'narr')

topics = {}

#copiado
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
    docs= {}
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


#CLUSTERING
#separar por hierarchical e partitioning
def clustering(D, approach, distance):
    clusters = [2,3,5,11,25,50]
    silhouettesaggt = []
    silhouettesaggd = []
    silhouetteskmt = []
    silhouetteskmd = []

    if approach == 'Agglomerative':
        if D == topics:
            for i in clusters:
                modelagg = AgglomerativeClustering(n_clusters=i).fit(vectorspace_topics.toarray())
                silhouettesaggt.append(silhouette_score(vectorspace_topics, modelagg.labels_, metric=distance))
            cluster = clusters[np.argmax(silhouettesaggt)]
            print(cluster)


        elif D == docs:
            for i in clusters:
                modelagg = AgglomerativeClustering(n_clusters=i).fit(vectorspace_dtrain.toarray())
                silhouettesaggd.append(silhouette_score(vectorspace_dtrain, modelagg.labels_, metric=distance))

    elif approach == 'K-means':
        if D == topics:
            for i in clusters:
                modelkmeans = KMeans(n_clusters=i).fit(vectorspace_topics.toarray())
                silhouetteskmt.append(silhouette_score(vectorspace_topics, modelkmeans.labels_, metric=distance))
        elif D == docs:
            for i in clusters:
                modelkmeans = KMeans(n_clusters=i).fit(vectorspace_dtrain.toarray())
                silhouetteskmd.append(silhouette_score(vectorspace_dtrain, modelkmeans.labels_, metric=distance))

    #comparing agg with kmeans
    print(max(silhouettesaggt))


#######não funciona - copiado do lab
def plot_dendrogram(model, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
########################################################

#INTERPRET
def interpret(cluster,D):
#descrever cluster considerando median and medoid criteria
#1º arranjar o cluster


#EVALUATE
#def evaluate(D):
#@behavior evaluates a solution produced by the introduced clustering function
#@output clustering internal( and optionally external) criteria


def main():
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = setup()

    #não percebo nada disto, perguntar
    global vectorspace_topics, vectorspace_dtrain
    tfidf_vectorizer = TfidfVectorizer(use_idf=False)
    vectorspace_topics = tfidf_vectorizer.fit_transform(topics)
    vectorspace_dtrain = tfidf_vectorizer.fit_transform(docs)


    #agglomerative_clustering(docs, 'Agglomerative', 'cosine')
    #clustering(topics, 'Agglomerative', 'euclidean')
    #clustering(topics, 'K-means', 'euclidean')
    return 0


if __name__ == '__main__':
    main()


