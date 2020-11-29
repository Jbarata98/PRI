import itertools
import collections
import networkx as nx
import pagerank as pk
from copy import deepcopy
import jsonpickle
from statistics import mean
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

import classifier as cl
import main as p1

from metrics import *

from parsers import *

from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances

threshold = 0.4
SUBSET_SIZE = 1000
CLASSIFIER = cl.mlp_classifier
VECTORIZER = cl.tfidf_vectorizer


def compute_similarity_matrix(d, similarity, th):
    tfidf_vectorizer = cl.tfidf_vectorizer
    matrix = tfidf_vectorizer.fit_transform(' '.join(list(value.values())) for key, value in d.items())
    df = pd.DataFrame(matrix.todense(),
                      index=d.keys())

    sim_matrix = np.array(similarity(df, df))
    for row in sim_matrix:
        row[row < th] = 0  # removes edges that dont meet the threshold
    # print({'matrix shape:', sim_matrix.shape})
    return sim_matrix


def build_graph(D, sim, th):
    D = dict(itertools.islice(D.items(), SUBSET_SIZE)) if len(D) > 1000 else D
    sim_matrix = compute_similarity_matrix(D, sim, th)  # calculated the similarities into a matrix
    sim_graph = nx.from_numpy_matrix(sim_matrix)  # transforms into graph
    mapping_dict = {node: doc_id for node, doc_id in zip(sim_graph.nodes, D)}
    sim_graph = nx.relabel_nodes(sim_graph, mapping_dict)
    # print({'nr of edges': sim_graph.number_of_edges()})
    sim_graph.remove_edges_from(nx.selfloop_edges(sim_graph))  # removes self loop edges

    # print(sim_graph.edges.data())

    # nx.draw(sim_graph)
    # plt.show()
    return sim_graph


def undirected_page_rank(q, D, p, sim, th, baseline = False):
    results = {}
    doc_ids = q['visited_documents']
    sim_graph = build_graph(cl.get_subset(D, doc_ids), sim, th)
    if baseline:
        pr_values = {'vanilla_pk': pk.pagerank(sim_graph, max_iter=50, weight=None)}
    else:
        pr_values = {'vanilla_pk': pk.pagerank(sim_graph, max_iter=50, weight=None),
                    'extended_pk': pk.pagerank(sim_graph, max_iter=50, weight='weight',
                                            personalization=q['document_probabilities'])}

    for pr_type in pr_values:
        results[pr_type] = cl.get_subset(pr_values[pr_type],
                                         sorted(list(pr_values[pr_type].keys()), key=lambda x: pr_values[pr_type][x],
                                                reverse=True)[:p])
    return results


def classify_graph(classification_results, Dtest, Qtest, Rtest, type, th, base = False):
    ranking_results = {q_id: {'related_documents': set(doc_ids)} for q_id, doc_ids in Rtest['p'].items()}
    pk_type = 'vanilla_pk' if type == 'vanilla' else 'extended_pk'
    with cl.tqdm(Qtest, desc=f'{f"CLASSIFYING {list(Qtest)[0]}":20}', leave=False) as q_tqdm:
        for q in q_tqdm:
            q_tqdm.set_description(desc=f'{f"CLASSIFYING {q}":20}')

            retrieved_docs_ids = \
            undirected_page_rank(classification_results[q], D=docs['test'], p=-1, sim=cosine_similarity, th=threshold, baseline = base)[
                pk_type]

            ranking_result = {

                # 'unrelated_documents': classification_results[q]['unrelated_documents'] ,
                'total_result': len(retrieved_docs_ids),
                'visited_documents': list(retrieved_docs_ids),
                'visited_documents_orders': {doc_id: rank + 1 for rank, doc_id in enumerate(retrieved_docs_ids)},
                'assessed_documents': {doc_id: (rank + 1, int(doc_id in Rtest['p'].get(q, []))) for rank, doc_id in
                                       enumerate(retrieved_docs_ids) if
                                       doc_id in Rtest['p'].get(q, []) or doc_id in Rtest['n'].get(q, [])},
                'document_probabilities': retrieved_docs_ids
            }
            ranking_results[q].update(ranking_result)
    return ranking_results


def get_pk_results(classification_baseline, Dtest, Qtest, Rtest, type, threshold):
    # <Get Retrieval results>
    pk_results_file = f"pagerank_results/eval_{type}_{threshold}"
    if not os.path.exists("pagerank_results"):
        os.mkdir("pagerank_results")
    if os.path.isfile(pk_results_file):
        print(f"Retrieval results already exist, loading from file (\"{pk_results_file}\")...")
        pk_results = jsonpickle.decode(open(pk_results_file, encoding='ISO-8859-1').read())
    else:
        print(f"Retrieval results don't exist, retrieving with model...")
        pk_results = classify_graph(classification_baseline, Dtest, Qtest, Rtest, type=type, th=threshold)
        with open(pk_results_file, 'w', encoding='ISO-8859-1') as f:
            f.write(jsonpickle.encode(pk_results, indent=4))
    # </Get Retrieval results>
    return pk_results


def plot_variation_threshold(classification_results, docs, topics, topic_index, pk_type):
    th_scores = defaultdict(dict)
    th_scores[f"Baseline {CLASSIFIER}"].update(classification_results)
    for th in np.arange(0.0, 1.0, 0.1):
        th_scores[f"{th:.1f} threshold"].update(
            get_pk_results(classification_results, docs, topics, topic_index, type=pk_type, threshold=th))
    return th_scores


def plot_avg_centrality(D, sim, th):
    graph = build_graph(D, sim, th)
    node_degrees = nx.degree_centrality(graph)
    top = 10
    avg_degree = mean(node_degrees[k] for k in node_degrees)
    print(f"Top {top} node (document) centralities:", collections.Counter(node_degrees).most_common(50))
    print(f"Average node (document) centralities:", avg_degree)


def plot_statistics_for_graph(graph_results, pk_type):
    metrics_per_sorted_topic(graph_results, title=f'{pk_type} PageRank metrics per topic')
    print_general_stats(graph_results, title='f{pk_type} PageRank metrics per topic')

def compare_graph_to_baseline(Dtest,Qtest,Rtest,threshold):
    try:
        p1_ranking = p1.evaluation(topics, (doc_index['p'], doc_index['n']), ('test', docs['test']),
                                   (p1.stem_analyzer,), (p1.NamedBM25F(K1=2, B=1),), skip_indexing=True)
    except Exception as e:
        p1_ranking = p1.evaluation(topics, (doc_index['p'], doc_index['n']), ('test', docs['test']),
                                   (p1.stem_analyzer,), (p1.NamedBM25F(K1=2, B=1),), skip_indexing=False)
    p1_ranking = list(p1_ranking.values())[0]


    pk_results_vanilla = classify_graph(p1_ranking, Dtest, Qtest, Rtest, type='vanilla', th=threshold, base = True)
    plot_iap_for_models({'Baseline IR System': p1_ranking, 'Vanilla': pk_results_vanilla})



QUESTION = 'a'
FUNCTIONALITY = None


def main():
    global docs, topics, topic_index, doc_index
    docs, topics, topic_index, doc_index = cl.setup()
    cl.docs = docs

    classification_results = cl.get_classification_results(docs['test'], topics, topic_index,
                                                           classifier=CLASSIFIER, vectorizer=VECTORIZER)
    threshold = 0.5
    if FUNCTIONALITY == 'a':
        graph = build_graph(docs['test'], sim=cosine_similarity(), th=threshold)
        print("graph:", graph)

    elif FUNCTIONALITY == 'b':
        for topic in classification_results:
            pr_values = undirected_page_rank(classification_results[topic], D=docs['test'], p=10, sim=cosine_similarity,
                                             th=threshold)
        print({topic: pr_values})

    elif QUESTION == 'a':

        compare_graph_to_baseline(docs['test'], topics, topic_index, threshold)

        graph_results_vanilla = get_pk_results(classification_results, docs['test'], topics, topic_index,
                                               type='vanilla', threshold = threshold)


        plot_statistics_for_graph(graph_results_vanilla, pk_type='Vanilla')

    elif QUESTION == 'b':
        print("Question b: \n")
        th_variation = plot_variation_threshold(classification_results, docs['test'], topics, topic_index,
                                                pk_type='extended_pk')
        plot_iap_for_models(th_variation)

    elif QUESTION == 'c':
        print("Question c: \n")
        plot_avg_centrality(docs['test'], sim=cosine_similarity, th=threshold)

    elif QUESTION == 'd':
        graph_results_vanilla = get_pk_results(classification_results, docs['test'], topics, topic_index,
                                               type='vanilla', threshold=threshold)
        graph_results_personalized = get_pk_results(classification_results, docs['test'], topics, topic_index,
                                                    type='extended', threshold=threshold)
        plot_iap_for_models({'vanilla': graph_results_vanilla, 'personalized': graph_results_personalized})


if __name__ == '__main__':
    main()
