import numpy as np
from metrics import *
from parsers import *
import classifier as cl
import itertools

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cdist


SUBSET_SIZE = 1000

#CLUSTERING
#separar por hierarchical e partitioning
def clustering(D, approach, distance):
    clusters = [300]
    distortions, silhouettes = [], []
    D = dict(itertools.islice(D.items(),SUBSET_SIZE)) if len(D) > 1000 else D
    tfidf_vectorizer = cl.tfidf_vectorizer
    vector_space =  tfidf_vectorizer.fit_transform(' '.join(list(value.values())) for key,value in D.items())
    df = pd.DataFrame(vector_space.todense(),
                      index=D.keys())
    print(list(df.shape))
    X = vector_space.toarray()
    if approach == 'Kmeans':
        for nr in clusters:
            model = KMeans(n_clusters=nr, verbose=1).fit(X)
            cluster_labels  = model.labels_
            silhouettes.append(silhouette_score(vector_space, cluster_labels, "cosine"))
            print(silhouettes)
            distortions.append(sum(np.min(cdist(X, model.cluster_centers_, "cosine"), axis = 1)) / vector_space.toarray().shape[0])

       # plt.plot(clusters, distortions, color='blue', label="elbow", linestyle='--') #elbow method
        plt.plot(clusters, silhouettes, color='red', label="silhouette", linestyle='--') #silhouette
        plt.xlabel("k")
        plt.ylabel("Silhouettes")
        plt.title("Silhouette Scores showing the optimal k")
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

    #comparing agg with kmeans


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
    return 0

#EVALUATE
#def evaluate(D):
#@behavior evaluates a solution produced by the introduced clustering function
#@output clustering internal( and optionally external) criteria


def main():
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = cl.setup()

    #agglomerative_clustering(docs, 'Agglomerative', 'cosine')
    clustering(docs['train'], 'Kmeans', 'manhattan')
    #clustering(topics, 'K-means', 'euclidean')
    return 0


if __name__ == '__main__':
    main()


