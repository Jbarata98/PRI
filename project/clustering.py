import numpy as np
from kneed import KneeLocator

from metrics import *
from parsers import *
import classifier as cl
import itertools

from time import time
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score


SUBSET_SIZE = 1000
RANDOM_STATE = 123
DEFAULT_N_INIT = 2


def checkpoint():
    start = time()
    temp = 0
    while True:
        temp = start
        start = time()
        yield f"{start - temp:.2f}s"


# CLUSTERING
def clustering(D, approach, distance):
    global model
    cluster_output = []
    cp = checkpoint()
    clusters = list([int(1.45 ** i) for i in range(2, 25)]) if SUBSET_SIZE > 1000 else range(10,1000,100)
    distortions, silhouettes, inertias = [], [], []
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
            inertias.append(model.inertia_)
            distortions.append(sum(np.min(pairwise_distances(X, model.cluster_centers_, "cosine"), axis=1)) /
                               vector_space.toarray().shape[0])
            print("Distortions (sparse):", next(cp))
        elif approach == 'Agglomerative':
            model = AgglomerativeClustering(n_clusters=nr, affinity=distance, linkage="single").fit(X.toarray())
        elif approach == 'Ward':
            model = AgglomerativeClustering(n_clusters=nr, affinity=distance, linkage="ward").fit(X.toarray())

        cluster_labels = model.labels_
        print(f"{approach} {nr}:", next(cp))

        silhouettes.append(silhouette_score(vector_space.toarray(), cluster_labels, "euclidean"))
        print("Silhouettes:", next(cp))
        print(silhouettes)
    if approach == 'Kmeans':
        KneeLocator(clusters, inertias, curve = "convex", direction = "decreasing").plot_knee()
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.title(f"{approach} Inertia Scores showing the optimal k with distance {distance}")
        plt.show()

        KneeLocator(clusters, distortions, curve = "convex", direction = "decreasing").plot_knee()
        plt.xlabel("k")
        plt.ylabel("Distortions")
        plt.title(f"{approach} Distortions Scores showing the optimal k with distance {distance}")
        plt.show()


    KneeLocator(clusters, silhouettes, curve = "concave", direction = "increasing", online= True).plot_knee()
    kneel = KneeLocator(clusters, silhouettes, curve = "concave", direction = "increasing", online= True)

    plt.xlabel("k")
    plt.ylabel("Silhouettes")
    plt.title(f"{approach} Silhouette Scores showing the optimal k with distance {distance}")
    plt.show()
    print('Optimal K is:', kneel.knee)
    if approach == 'Kmeans':
        model = KMeans(n_clusters=kneel.knee, verbose=1, random_state=RANDOM_STATE, n_init=DEFAULT_N_INIT).fit(X)
        mydict = {i: np.where(model.labels_ == i)[0] for i in range(model.n_clusters)}

        # Transform the dictionary into list
        dictlist = []
        for key, value in mydict.items():
            temp = (key, value)
            dictlist.append(temp)
        return dictlist
    if approach == 'Agglomerative':
        return 0




# INTERPRET
def interpret(cluster, D):
    tfidf_vectorizer = cl.tfidf_vectorizer
    vector_space = tfidf_vectorizer.fit_transform(' '.join(list(value.values())) for key, value in D.items())
    # median - relevance of each term for the documents in the cluster
    for i in cluster:
        np.median(i[0], axis=0)

    #medoid -  object within a cluster for which average dissimilarity between it and all the other the members of the cluster is minimal, most central document in the cluster
    dist_mat = pairwise_distances(vector_space, cluster, "cosine")
    medoid = np.argmin(dist_mat.sum(axis=0))
    print(medoid)


# EVALUATE
def evaluate(D, cluster):
    tfidf_vectorizer = cl.tfidf_vectorizer
    vector_space = tfidf_vectorizer.fit_transform(' '.join(list(value.values())) for key, value in D.items())
    #External Measures:
    # Rand Index - computes a similarity measure between two clusterings
    labels_true = cluster
    labels_pred = D.target
    rand_score = adjusted_rand_score(labels_true, labels_pred)
    print('adjusted rand score = {}'.format(rand_score))

    #Internal Measures:
    # Silhouette Coefficient
    sil_score = silhouette_score(vector_space, cluster, metric='cosine')
    print('silhouette score = {}'.format(sil_score))

FUNCTIONALITY = 'a'

def main():
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = cl.setup()
    if FUNCTIONALITY == 'a':
        cluster = clustering(docs['train'], 'Kmeans', 'cosine')
        print("cluster output:", cluster)
    if FUNCTIONALITY == 'b':
        interpret(cluster, docs['train'])
    if FUNCTIONALITY == 'c':
        evaluate(docs['train'], cluster)


    return 0


if __name__ == '__main__':
    main()
