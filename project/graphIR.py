import itertools
import collections
import networkx as nx
import pagerank as pk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

import classifier as cl
from metrics import *
from parsers import *

from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances

threshold = 0.4
SUBSET_SIZE = 1000
top_p = 10

def compute_similarity_matrix(d, similarity,th):
    tfidf_vectorizer = cl.tfidf_vectorizer
    matrix = tfidf_vectorizer.fit_transform(' '.join(list(value.values())) for key,value in d.items())
    df = pd.DataFrame(matrix.todense(),
                      index= d.keys())

    sim_matrix = np.array(similarity(df,df))
    for row in sim_matrix:
        row[row < th] = 0  # removes edges that dont meet the threshold
    # print({'matrix shape:', sim_matrix.shape})
    return sim_matrix


def build_graph(D,sim,th):
    D = dict(itertools.islice(D.items(),SUBSET_SIZE)) if len(D) > 1000 else D # 1000 docs subset
    sim_matrix = compute_similarity_matrix(D,sim,th) # calculated the similarities into a matrix
    sim_graph = nx.from_numpy_matrix(sim_matrix) # transforms into graph
    mapping_dict = {node: doc_id for node,doc_id in zip(sim_graph.nodes,D)}
    sim_graph = nx.relabel_nodes(sim_graph, mapping_dict)
    # print({'nr of edges': sim_graph.number_of_edges()})
    sim_graph.remove_edges_from(nx.selfloop_edges(sim_graph)) # removes self loop edges

    #print(sim_graph.edges.data())

    # nx.draw(sim_graph)
    # plt.show()
    return sim_graph

def undirected_page_rank(q,D,p,sim,th):
    results = {}
    doc_ids = q['related_documents']
    sim_graph = build_graph(cl.get_subset(D,doc_ids), sim, th)

    pr_values = {'vanilla_pk' : pk.pagerank(sim_graph, max_iter=50, weight=None),
                 'extended_pk': pk.pagerank(sim_graph, max_iter=50, weight='weight', personalization = q['document_probabilities'])}

    for pr_type in pr_values:
        results[pr_type] = cl.get_subset(pr_values[pr_type],sorted(list(pr_values[pr_type].keys()),key=lambda x:pr_values[pr_type][x])[:p:-1])
    return results

def main():
    D = {}
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = cl.setup()
    classification_results = cl.get_classification_results(docs['test'],topics, topic_index, classifier = cl.knn_classifier, vectorizer = cl.tfidf_vectorizer)
    for topic in classification_results:
        pr_values = undirected_page_rank(classification_results[topic], docs['test'], top_p, sim = cosine_similarity, th = threshold)
        print({topic : pr_values})

if __name__ == '__main__':
    main()