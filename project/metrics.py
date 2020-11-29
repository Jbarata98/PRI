import math
from collections import defaultdict

import matplotlib.pyplot as plt
import ml_metrics
import numpy as np
import pandas as pd
import seaborn as sns

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
BETA_SQR = 0.5 ** 2
# K_TESTS = tuple(range(1, DEFAULT_P + 1))
K_TESTS = tuple([int(1.5 ** i) for i in range(1, 18)]) + (1000,)
# K_TESTS = (1, 3, 5, 10, 20, 50, 100, 200, 500, DEFAULT_P)
#
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
        if show_points:
            ax.scatter(x, y, 15, alpha=0.5)
        legend.append(name)
    ax.legend(legend, ncol=(len(legend) // 4 + 1), loc='best', fancybox=True, shadow=True, borderaxespad=0)
    ax.tick_params(axis='x', labelrotation=min(len(x), 90))


def calc_precision_based_measures(predicted_ids, expected_ids, ks=(10,), metric=None):
    def precision(_predicted, _expected, k):
        return len(set(_predicted[:k]).intersection(set(_expected))) / len(_predicted[:k])

    def recall(_predicted, _expected, k):
        return len(set(_predicted[:k]).intersection(set(_expected))) / len(_expected)

    def fbeta(_predicted, _expected, k):
        pre, rec = precision(_predicted, _expected, k), BETA * recall(_predicted, _expected, k)
        return 0.0 if pre == rec == 0 else (BETA_SQR + 1) * pre * rec / (BETA_SQR * pre + rec)

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

    return {f'{measure}@{k}': metrics[measure]([int(i) for i in predicted_ids], [int(i) for i in expected_ids], k) for k
            in ks for measure in metric}


def precision_recall_generator(predicted, expected):
    tp = 0
    for i, id in enumerate(predicted):
        if id in expected:
            tp += 1
        yield tp / (i + 1), tp / max(1, len(expected))


def tp_fp_fn_generator(predicted, expected):
    tp = 0
    for i, id in enumerate(predicted):
        if id in expected:
            tp += 1
        yield tp, i + 1 - tp, len(expected) - tp


def plot_tp_fp_fn_for_p(ranking_results):
    tp_fp_fn = {'true positives': defaultdict(list), 'false positives': defaultdict(list),
                'false negatives': defaultdict(list)}

    for q_id, data in ranking_results.items():
        tp_fp_fn_g = tp_fp_fn_generator(data['visited_documents'], data['related_documents'])
        for tp, fp, fn in tp_fp_fn_g:
            tp, fp, fn = np.array([tp, fp, fn]) / sum([tp, fp, fn])
            tp_fp_fn['true positives'][q_id].append(tp)
            tp_fp_fn['false positives'][q_id].append(fp)
            tp_fp_fn['false negatives'][q_id].append(fn)

    for metric, data in tp_fp_fn.items():
        tp_fp_fn[metric] = np.mean(list(data.values()), axis=0)

    multiple_line_chart(plt.gca(), list(range(1, len(list(tp_fp_fn.values())[0]) + 1)), tp_fp_fn,
                        'True positives, false positives and false negatives false for p', 'p', 'score',
                        False, False, False)
    plt.show()


def plot_precicion_recall_for_p(ranking_results):
    precision_recall = {'precision': defaultdict(list), 'recall': defaultdict(list)}

    for q_id, data in ranking_results.items():
        precision_recall_g = precision_recall_generator(data['visited_documents'], data['related_documents'])
        for precision, recall in precision_recall_g:
            precision_recall['precision'][q_id].append(precision)
            precision_recall['recall'][q_id].append(recall)

    for metric, data in precision_recall.items():
        precision_recall[metric] = np.mean(list(data.values()), axis=0)

    multiple_line_chart(plt.gca(), list(range(1, len(list(precision_recall.values())[0]) + 1)), precision_recall,
                        'Precision and recall for p', 'p', 'score',
                        False, False, True)
    plt.show()


def calc_gain_based_measures(predicted, expected, k_values=(5, 10, 15, 20), metric=None):
    def ndcg(predicted, expected):
        sum_dcg, sum_ndcg = 0, 0
        binary_relevance, dcg, optimal_dcg, ndcg = [], [], [], []
        for i, id in enumerate(predicted):
            if id in expected:
                sum_dcg += 1 / math.log(i + 2, 2)
                binary_relevance.append(1)
            else:
                binary_relevance.append(0)
            if (i + 1) in k_values:
                dcg.append(sum_dcg)

        for j, value in enumerate(sorted(binary_relevance, reverse=True)):
            sum_ndcg += value / math.log(j + 2, 2)
            if (j + 1) in k_values:
                optimal_dcg.append(sum_ndcg)
        ndcg = [dcg[k] / optimal_dcg[k] if optimal_dcg[k] else 0 for k in range(len(dcg))]
        return ndcg + ndcg[-1:] * (len(k_values) - len(dcg))  # adjust for missing k values

    metrics = {
        'nDCG': ndcg,
    }

    if metric is None:
        metric = metrics.keys()

    return {measure: metrics[measure](predicted, expected) for measure in
            metric}


def MRR(predicted, expected):
    MRR = 0
    for i, qid in zip(range(MaxMRRRank), predicted):
        if qid in expected:
            MRR += 1 / (i + 1)
            break
    return {'MRR': MRR}


def BPREF(predicted, relevant, non_relevant):
    relevant_answers, non_relevant_answers = set(predicted).intersection(set(relevant)), set(predicted).intersection(
        set(non_relevant))
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


def precision_boolean_metrics(I, retrieval_results):
    doc_ids_total = list(I.D.keys())
    print(len(doc_ids_total))
    confusion_matrix_vals = defaultdict(int)
    for q_id in retrieval_results.keys():
        unrelated_documents = set(doc_ids_total).difference(retrieval_results[q_id]['related_documents'])
        unvisited_documents = set(retrieval_results[q_id]['related_documents']).union(unrelated_documents)
        confusion_matrix_vals['tp'] += len(
            set([k for k, v in retrieval_results[q_id]['assessed_documents'].items() if v]))
        confusion_matrix_vals['tn'] += len(unvisited_documents.intersection(unrelated_documents))
        confusion_matrix_vals['fp'] += len(
            set(retrieval_results[q_id]['assessed_documents']).intersection(unrelated_documents))
        confusion_matrix_vals['fn'] += len(
            unvisited_documents.intersection(set(retrieval_results[q_id]['related_documents'])))
    confusion_matrix_vals['precision'], confusion_matrix_vals['recall'], confusion_matrix_vals['f-beta'] = np.mean(
        confusion_matrix_vals['precision']), np.mean(
        confusion_matrix_vals['recall']), np.mean(confusion_matrix_vals['f-beta'])
    return confusion_matrix_vals


def bar_chart(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, percentage=False,
              reverse=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    if reverse is not None:
        yvalues, xvalues = zip(*sorted(zip(yvalues, xvalues), reverse=reverse))
    ax.set_xticklabels(xvalues, rotation=90, fontsize='small')
    ax.bar(xvalues, yvalues, edgecolor='grey')
    plt.show()


def calculate_precision_boolean(I, retrieval_results, normalized=False):
    doc_ids_total = list(I.D.keys())
    precision_dict = defaultdict(list)
    for q_id in retrieval_results.keys():
        unrelated_documents = set(doc_ids_total).difference(retrieval_results[q_id]['related_documents'])
        unvisited_documents = set(retrieval_results[q_id]['related_documents']).union(unrelated_documents)
        tp = len(set([k for k, v in retrieval_results[q_id]['assessed_documents'].items() if v]))
        fp = len(set(retrieval_results[q_id]['assessed_documents']).intersection(unrelated_documents))
        fn = len(unvisited_documents.intersection(set(retrieval_results[q_id]['related_documents'])))
        precision = tp / (tp + fp) if tp else 0
        recall = tp / (tp + fn) if tp else 0
        precision_dict['precision'].append(0 if not tp else tp / (tp + fp))
        precision_dict['recall'].append(0 if not tp else tp / (tp + fn))

        precision_dict['f-beta'].append(0 if precision == recall == 0 else ((1 + 0.5 ** 2) * (precision * recall)) / (
                    (0.5 ** 2) * precision + recall))

    if normalized:
        precision_dict['precision'], precision_dict['recall'], precision_dict['f-beta'] = np.mean(
            precision_dict['precision']), np.mean(precision_dict['recall']), np.mean(precision_dict['f-beta'])
    return precision_dict


def print_confusion_matrix(retrieval_results):
    total = sum(list(retrieval_results.values()))
    cm = {tag: val / total for tag, val in retrieval_results.items()}
    ls = ["Relevant", "Irrelevant"]
    cm_array = np.array([[cm['tp'], cm['fp']], [cm['fn'], cm['tn']]])
    plot_confusion_matrix(plt.gca(), cm_array, ls, title="Confusion Matrix Boolean", class_name="Label")
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=ls)
    # disp.plot()
    plt.show()


def plot_confusion_matrix(ax, cnf_mtx, labels, title="", class_name="class"):
    data = pd.DataFrame(cnf_mtx)
    ax.set_title(title)
    ax.set_xlabel(class_name)
    ax.set_ylabel(class_name)
    sns.heatmap(data,
                xticklabels=labels,
                yticklabels=labels,
                square=True,
                cbar=False,
                annot=True,
                cmap='Blues',
                linewidths=1,
                linecolor="black",
                fmt='g',
                ax=ax)
    return


def print_general_stats(precision_results, title=None):
    metrics_scores, results = defaultdict(list), defaultdict(list)

    for q_id, data in precision_results.items():
        for metric, score in calc_precision_based_measures(data['visited_documents'], data['related_documents'],
                                                           K_TESTS).items():
            metrics_scores[metric].append(score)
        for i, score in enumerate(
                calc_gain_based_measures(data['visited_documents'], data['related_documents'], K_TESTS)['nDCG']):
            metrics_scores[f'NDCG@{K_TESTS[i]}'].append(score)

    for metric, scores in metrics_scores.items():
        metrics_scores[metric] = np.mean(scores)

    for metric, score in metrics_scores.items():
        results[metric.split('@')[0]].append(score)

    results['BPref'] = [stats['value'] for k, stats in ir.bpref(precision_results, K_TESTS[:-1] + ('all',)).items()]
    results['BPref'] = [x if x else results['BPref'][-1] for x in results['BPref']]
    multiple_line_chart(plt.gca(), list(K_TESTS), results, 'Metrics' + (f" for {title}" if title else ""), 'k', 'score',
                        True, False, True)
    plt.show()


def metrics_per_sorted_topic(precision_results, title=None):
    metrics_scores = defaultdict(list)
    map_results = {}

    for q_id, data in precision_results.items():
        metric, score = list(
            calc_precision_based_measures(data['visited_documents'], data['related_documents'], (DEFAULT_P,),
                                          ('map',)).items())[0]
        map_results[q_id] = score

    sorted_q_ids = sorted(map_results.keys(), key=map_results.get)

    for q_id in sorted_q_ids:
        data = precision_results[q_id]
        for metric, score in calc_precision_based_measures(data['visited_documents'], data['related_documents'],
                                                           (10, DEFAULT_P),
                                                           ('precision', 'recall', 'fbeta', 'map')).items():
            metrics_scores[metric].append(score)
        metrics_scores['BPref'].append(list(ir.bpref({q_id: precision_results[q_id]}, (10,)).items())[0][1]['value'])

    picked = {'BPref@10': metrics_scores['BPref'], 'MAP': metrics_scores[f'map@{DEFAULT_P}'],
              'precision@10': metrics_scores[f'precision@{10}'], 'recall@10': metrics_scores[f'recall@{10}']}

    multiple_line_chart(plt.gca(), sorted_q_ids, picked,
                        'Metrics by topic, sorted by MAP score' + f" for {title}" if title else "", 'topic', 'score',
                        True, False, True)
    plt.show()


def plot_iap_for_models(models_ranking_results):
    Y = {}
    for model, ranking_results in models_ranking_results.items():
        iap = ir.iap(ranking_results)
        x, Y[model] = zip(*(iap.items()))
        x = [float(val) for val in x]
    multiple_line_chart(plt.gca(), x, Y, 'Eleven Point - Interpolated Average Precision (IAP)', 'recall', 'precision',
                        False, True, True)
    plt.show()
