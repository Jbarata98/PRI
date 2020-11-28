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
    D = dict(itertools.islice(D.items(),SUBSET_SIZE)) if len(D) > 1000 else D
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
        results[pr_type] = cl.get_subset(pr_values[pr_type],sorted(list(pr_values[pr_type].keys()),key=lambda x:pr_values[pr_type][x], reverse = True)[:p])
    return results

def classify_graph(classification_results, Dtest, Qtest, Rtest, type):
    ranking_results = {q_id: {'related_documents': set(doc_ids)} for q_id, doc_ids in Rtest['p'].items()}
    pk_type = 'vanilla_pk' if type == 'vanilla' else 'extended_pk'
    with cl.tqdm(Qtest, desc=f'{f"CLASSIFYING {list(Qtest)[0]}":20}', leave=False) as q_tqdm:
        for q in q_tqdm:
            retrieved_docs_ids = undirected_page_rank(classification_results[q], D = docs['test'], p = -1, sim = cosine_similarity, th = threshold)[pk_type]

            ranking_result = {
                'unrelated_documents': classification_results[q]['unrelated_documents'],
                'total_result': len(retrieved_docs_ids),
                'visited_documents': list(retrieved_docs_ids),
                'visited_documents_orders': {doc_id: rank + 1 for rank, doc_id in enumerate(retrieved_docs_ids)},
                'assessed_documents': {doc_id: (rank + 1, int(doc_id in Rtest['p'].get(q, []))) for rank, doc_id in enumerate(retrieved_docs_ids) if
                                       doc_id in Rtest['p'].get(q, []) or doc_id in Rtest['n'].get(q, [])},
                'document_probabilities': retrieved_docs_ids
            }
            ranking_results[q].update(ranking_result)
    return ranking_results



def main():
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = cl.setup()
    classification_results = cl.get_classification_results(docs['test'],topics, topic_index, classifier = cl.knn_classifier, vectorizer = cl.tfidf_vectorizer)
    for topic in classification_results:
        pr_values = undirected_page_rank(classification_results[topic], D = docs['test'], p = 10, sim = cosine_similarity, th = threshold)
        # print({topic : pr_values})
    graph_results_vanilla  = classify_graph(classification_results, docs['test'],topics,topic_index, type = 'vanilla'),
    graph_results_personalized = classify_graph(classification_results, docs['test'], topics, topic_index, type='extended')

    metrics_per_sorted_topic(graph_results_personalized, title = 'Vanilla PageRank metrics per topic')
    print_general_stats(graph_results_personalized, title = 'Vanilla PageRank metrics per topic')
    plot_iap_for_models({'vanilla': graph_results_vanilla, 'personalized': graph_results_personalized})

if __name__ == '__main__':
    main()