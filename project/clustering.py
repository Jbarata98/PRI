import numpy as np
from metrics import *
from parsers import *
import classifier as cl
import itertools

from time import time
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cdist

SUBSET_SIZE = 10000
RANDOM_STATE = 123
DEFAULT_N_INIT = 3


def checkpoint():
    start = time()
    temp = 0
    while True:
        temp = start
        start = time()
        yield f"{start - temp:.2f}s"


# CLUSTERING
# separar por hierarchical e partitioning
def clustering(D, approach, distance):
    cp = checkpoint()
    clusters = [300]
    distortions, silhouettes = [], []
    D = dict(itertools.islice(D.items(), SUBSET_SIZE)) if len(D) > 1000 else D
    tfidf_vectorizer = cl.tfidf_vectorizer
    print("Init:", next(cp))
    vector_space = tfidf_vectorizer.fit_transform(' '.join(list(value.values())) for key, value in D.items())
    print("TFIDF:", next(cp))
    df = pd.DataFrame(vector_space.todense(), index=D.keys())
    print(list(df.shape))
    X = vector_space# .toarray()    # much faster with sparse matrixes
    for nr in clusters:
        print(f"testing {approach} ... with {nr} clusters")

        if approach == 'Kmeans':
            model = KMeans(n_clusters=nr, verbose=1, random_state=RANDOM_STATE, n_init=DEFAULT_N_INIT).fit(X)
        elif approach == 'Agglomerative':
            model = AgglomerativeClustering(n_clusters=nr, affinity=distance, linkage="complete").fit(X.toarray())


        # distortions.append(sum(np.min(pairwise_distances(X, model.cluster_centers_, "cosine"), axis=1)) / vector_space.toarray().shape[0])
        # print("Distortions (sparse):", next(cp))
        # distortions.append(sum(np.min(cdist(X, model.cluster_centers_, "cosine"), axis=1)) / vector_space.toarray().shape[0])
        # print("Distortions (dense):", next(cp))

        cluster_labels = model.labels_
        print(f"{approach} {nr}:", next(cp))

        silhouettes.append(silhouette_score(vector_space, cluster_labels, "cosine"))
        print("Silhouettes:", next(cp))
        print(silhouettes)

        # plt.plot(clusters, distortions, color='blue', label="elbow", linestyle='--') #elbow method
    plt.plot(clusters, silhouettes, color='red', label="silhouette", linestyle='--')  # silhouette
    plt.xlabel("k")
    plt.ylabel("Silhouettes")
    plt.title(f"{approach} Silhouette Scores showing the optimal k ")
    plt.show()

        # silhouettesaggt.append(silhouette_score(vectorspace_topics, modelagg.labels_, metric=distance))
        # cluster = clusters[np.argmax(silhouettesaggt)]
        # print(cluster)

    #
    #     elif D == docs:
    #         for i in clusters:
    #             modelagg = AgglomerativeClustering(n_clusters=i).fit(vectorspace_dtrain.toarray())
    #             silhouettesaggd.append(silhouette_score(vectorspace_dtrain, modelagg.labels_, metric=distance))
    #
    # elif approach == 'K-means':
    #     if D == topics:
    #         for i in clusters:
    #             modelkmeans = KMeans(n_clusters=i).fit(vectorspace_topics.toarray())
    #             silhouetteskmt.append(silhouette_score(vectorspace_topics, modelkmeans.labels_, metric=distance))
    #     elif D == docs:
    #         for i in clusters:
    #             modelkmeans = KMeans(n_clusters=i).fit(vectorspace_dtrain.toarray())
    #             silhouetteskmd.append(silhouette_score(vectorspace_dtrain, modelkmeans.labels_, metric=distance))

    # comparing agg with kmeans


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

# INTERPRET
def interpret(cluster, D):
    # descrever cluster considerando median and medoid criteria
    # 1º arranjar o cluster
    return 0


# EVALUATE
# def evaluate(D):
# @behavior evaluates a solution produced by the introduced clustering function
# @output clustering internal( and optionally external) criteria


def main():
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = cl.setup()

    # agglomerative_clustering(docs, 'Agglomerative', 'cosine')
    clustering(docs['train'], 'Agglomerative', 'cosine')
    # clustering(topics, 'K-means', 'euclidean')
    return 0


if __name__ == '__main__':
    main()
