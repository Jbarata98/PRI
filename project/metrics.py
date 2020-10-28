import math
import matplotlib.pyplot as plt
import ml_metrics
import numpy as np

from ir_evaluation.effectiveness import effectiveness


COLLECTION_LEN = 807168
COLLECTION_PATH = 'collection/'
DATASET = 'rcv1'
QRELS = 'qrels.test.txt'
TRAIN_DATE_SPLIT = '19960930'
AVAILABLE_DATA = ('headline', 'p', 'dateline', 'byline')
DATA_HEADER = ('newsitem', 'itemid')
TOPICS = "topics.txt"
TOPICS_CONTENT = ('num', 'title', 'desc', 'narr')
DEFAULT_K = 5
BOOLEAN_ROUND_TOLERANCE = 1 - 0.2
OVERRIDE_SAVED_JSON: bool = False
OVERRIDE_SUBSET_JSON = False
USE_ONLY_EVAL = True
MaxMRRRank = 10
EVAL_SAMPLE_SIZE = 3000
BETA = 0.5
K1_TEST_VALS = np.arange(0, 4.1, 0.5)
B_TEST_VALS = np.arange(0, 1.1, 0.2)
DEFAULT_P = 1000
K_TESTS = (1, 3, 5, 10, 20, 50, 100, 200, 500, DEFAULT_P)

ir = effectiveness()  # --> an object, which we can use all methods in it, is created


def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str,
                        show_points=False, xpercentage=False, ypercentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xpercentage:
        ax.set_xlim(0.0, 1.0)
    if ypercentage:
        ax.set_ylim(0.0, 1.0)

    x = xvalues
    for name, y in yvalues.items():
        if isinstance(xvalues, dict):
            x = xvalues[name]
        ax.plot(x, y)
        ax.scatter(x, y, 20, alpha=0.5)
        legend.append(name)
    ax.legend(legend, loc='best', fancybox=True, shadow=True, borderaxespad=0)


def calc_precision_based_measures(predicted_ids, expected_ids, ks=(10,), metric=None):
    def precision(_predicted, _expected, k):
        return len(set(_predicted[:k]).intersection(set(_expected))) / len(_predicted[:k])

    def recall(_predicted, _expected, k):
        return len(set(_predicted[:k]).intersection(set(_expected))) / len(_expected)

    def fbeta(_predicted, _expected, k):
        pre, rec = precision(_predicted, _expected, k), BETA * recall(_predicted, _expected, k)
        return 0.0 if pre == rec == 0 else 2 * pre * rec / (pre + rec)

    def map(_predicted, _expected, k):
        return ml_metrics.mapk([_expected], [_predicted], k)

    def MRR(predicted, expected, k):
        MRR = 0
        for i, qid in zip(range(k), predicted):
            if qid in expected:
                MRR = 1 / (i + 1)
                break
        return MRR

    metrics = {
        'precision': precision,
        'recall': recall,
        'fbeta': fbeta,
        'map': map,
        'mrr': MRR,
    }

    if metric is None:
        metric = metrics.keys()

    return {f'{measure}@{k}': metrics[measure]([int(i) for i in predicted_ids], [int(i) for i in expected_ids], k) for k in ks for measure in metric}


def precision_recall_generator(predicted, expected):
    tp = 0
    for i, id in enumerate(predicted):
        if id in expected:
            tp += 1
        yield tp / (i + 1), tp / max(1, len(expected))


def calc_gain_based_measures(predicted, expected, k_values = [5,10,15,20], metric=None):  # isto n deve tar bem
    def dcg(predicted, expected, k):
        sum_dcg, sum_ndcg = 0, 0
        binary_relevance, dcg, optimal_dcg, ndcg = [], [], [], []
        for i, id in enumerate(predicted):
            if id in expected:
                sum_dcg += 1 / math.log(i + 2, 2)
                binary_relevance.append(1)
            else:
                binary_relevance.append(0)
            if i in k_values:
                dcg.append(sum_dcg)

        for j, value in enumerate(sorted(binary_relevance, reverse=True)):
            sum_ndcg += value / math.log(j + 2, 2)
            if j in k_values:
                optimal_dcg.append(sum_ndcg)
        ndcg = [dcg[k] / optimal_dcg[k] for k in range(len(k_values))]
        return ndcg

    metrics = {
        'nDCG': dcg,
    }

    if metric is None:
        metric = metrics.keys()

    return {measure: metrics[measure](predicted, expected, k_values) for measure in
            metric}


def MRR(predicted, expected):
    MRR = 0
    for i, qid in zip(range(MaxMRRRank), predicted):
        if qid in expected:
            MRR += 1 / (i + 1)
            break
    return {'MRR': MRR}



def BPREF(predicted,relevant,non_relevant):
    relevant_answers,non_relevant_answers = set(predicted).intersection(set(relevant)),set(predicted).intersection(set(non_relevant))
    counter = 0
    Bpref = 0
    sum = 0
    if len(relevant_answers) == 0:
        return Bpref
    if len(non_relevant_answers) == 0:  # idk
        Bpref = 1 / len(relevant_answers) + 1
        return Bpref
    else:
        for rel in relevant_answers:
            for pred in predicted:
                if pred in non_relevant_answers:
                    counter += 1
                elif pred is rel:
                    sum += (1 - (counter / min(len(relevant_answers), len(non_relevant_answers))))
                    counter = 0
                    continue
        Bpref = 1 / len(relevant_answers) + sum

        return {'BPREF': Bpref}


def print_general_stats(precision_results, topic_index):
    print("Average Precision@n:")
    ap_at_n = ir.ap_at_n(precision_results, [5, 10, 15, 20, 'all'])
    print(ap_at_n)
    print("\n")
    print("R-Precision@n:")
    rprecision = ir.rprecision(precision_results, [5, 10, 15, 20, 'all'])
    print(rprecision)
    print("\n")
    print("Mean Average Precision:")
    mean_ap = ir.mean_ap(precision_results, [5, 10, 15, 20, 'all'])
    print(mean_ap)
    print("\n")
    print("F-Measure:")
    fmeasure = ir.fmeasure(precision_results, [5, 10, 15, 20, 'all'])
    print(fmeasure)
    print("\n")
    ########################################################################################
    # parameters -> (data, constant, boundaries)
    print("Geometric Mean Average Precision:")
    gmap = ir.gmap(precision_results, 0.3, [5, 10, 15, 20, 'all'])
    print(gmap)
    print("\n")
    ########################################################################################
    # parameters -> (data)
    print("Eleven Point - Interpolated Average Precision:")
    print("Recall => Precision")
    iap = ir.iap(precision_results)
    print(iap)
    X, Y = {}, {}
    X[''], Y[''] = zip(*(iap.items()))
    X[''] = [float(val) for val in X['']]
    multiple_line_chart(plt.gca(), X, Y, 'Eleven Point - Interpolated Average Precision (IAP)', 'recall', 'precision',
                        False, True, True)
    plt.show()


    # print("Normalized Discount Gain Measure:")
    # print("nDCG:")
    # ndcg = calc_gain_based_measures(precision_results[topic_index[1]]['visited_documents'], topic_index[1], k_values=list(range(1,len(precision_results[topic_index[1]]))))
    # print(ndcg)
    # multiple_line_chart(plt.gca(),list(range(1,len(precision_results[topic_index[1]]['visited_documents']))), ndcg, 'Normalized Discount Gain Measure', 'k', 'ndcg')
    #
    # plt.show()
    # #
    # print("Cumulative Gain:")
    # cgain = ir.cgain(precision_results, [5, 10, 15, 20, 'all'])
    # print(cgain)
    #
    # print("\n")
    #
    # print("Normalized Cumulative Gain:")
    # ncgain = ir.ncgain(precision_results, [5, 10, 15, 20])
    # print(ncgain)
    #
    # print("\n")
    #
    # print("Discounted Cumulative Gain:")
    # dcgain = ir.dcgain(precision_results, [5, 10, 15, 20])
    # print(dcgain)
    #
    # print("\n")
    #
    # print("Normalized Discounted Cumulative Gain:")
    # ndcgain = ir.ndcgain(precision_results, [5, 10, 15, 20, 'all'])
    # print(ndcgain)
    # parameters => (data, boundaries)
    print("BPref:")
    bpref = ir.bpref(precision_results, [5, 10, 15, 20, 'all'])
    print(bpref)
