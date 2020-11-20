import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import string

nodes = 5


def show_graph_with_labels(adjacency_matrix, mylabels=None):
    if mylabels is None:
        range(1, len(adjacency_matrix) + 1)
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    plt.show()
    return gr


def main():
    # am = np.array([np.random.choice([0, 1], size=(nodes,), p=[2. / 3, 1. / 3]) for _ in range(nodes)])
    am = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1]])
    G = show_graph_with_labels(am, {i: l for i, l in enumerate(string.ascii_uppercase[:len(am)])})
    print('1)', nx.pagerank(G, alpha=0.9))

    with open('pri_links.txt') as f:
        lines = f.readlines()
        n_l = 1240
        am = []
        for line in lines:
            ma_l = np.zeros(n_l)
            for adj in line.split():
                ma_l[int(adj) - 1] = 1
            am.append(ma_l)

    am = np.array(am)
    G = show_graph_with_labels(am)
    print('2)', nx.pagerank(G, alpha=0.9))


if __name__ == '__main__':
    main()
