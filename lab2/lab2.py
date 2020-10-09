from collections import defaultdict
from os import getcwd, listdir
from os.path import isfile, join
from math import log
from nltk import RegexpTokenizer
import numpy as np

import matplotlib.pyplot as plt


def get_count(file):
    dict = {}

    with open(file, encoding="utf8") as txt:
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(txt.read())

    for word in [word.lower() for word in words]:
        if word in dict.keys():
            dict[word] += 1
        else:
            dict[word] = 1
    return dict



def get_word_index(dir_name):
    inverted_list = defaultdict(list)
    file_path = getcwd() + "/" + dir_name
    for i, file_name in enumerate([f for f in listdir(file_path) if isfile(join(file_path, f))]):
        for word, count in get_count(dir_name + "/" + file_name).items():
            inverted_list[word].append((file_name.split('.')[0], count))
    return inverted_list


def terms_bar_plot(inverted_list):
    doc_terms = defaultdict(list)
    for word, indexes in inverted_list.items():
        for i, iid in enumerate(indexes):
            doc_terms[iid[0]] += [word]*iid[1]
    plt.figure(figsize=(10, 5))
    plt.hist(doc_terms.values(), bins=len(inverted_list), label=doc_terms.keys())
    plt.xticks(rotation=75)
    plt.legend()
    plt.show()
    return("<plot>")


def tokens_hist(inverted_list):
    doc_terms = defaultdict(int)
    for word, indexes in inverted_list.items():
        for i, iid in enumerate(indexes):
            doc_terms[iid[0]] += 1
    plt.figure(figsize=(10, 5))
    plt.hist(np.array(list(doc_terms.values())), bins=5)
    plt.show()


def idf(inverted_list):
    n_docs = len({item[0] for sublist in inverted_list.values() for item in sublist})
    return {word: log(n_docs/len(inverted_list[word])) for word in inverted_list}


def get_idf_stats(inverted_list):
    idf_stats, idfs = {}, idf(inverted_list)
    for word, indexes in inverted_list.items():
        idf_stats[word] = {'idf': idfs[word], 'indexes': indexes}
    return idf_stats


def ret_ocurrences(term, idf_stats):
    return [id[0] for id in idf_stats[term]['indexes']]


def ret_mul_ocurrences(terms, idf_stats):
    return [ret_ocurrences(term, idf_stats) for term in terms]


def similarity(query, idf_stats):
    n_docs = {item[0] for sublist in idf_stats for item in idf_stats[sublist]['indexes']}
    simmilarities = {word: 0 for word in n_docs}
    for term in query:
        for (id, freq) in idf_stats[term]['indexes']:
            simmilarities[id] += freq * idf_stats[term]['idf']
    return simmilarities

def main():
    print("1\n  1.1\n    a)")
    word_index = get_word_index("../lab1/brxts")
    for word, indexes in word_index.items():
        print(f'      {word:10}: {len(indexes):2}: {indexes}')
    print("    b)")
    print("  1.2\n", tokens_hist(word_index))
    idf_stats = get_idf_stats(word_index)
    print("  1.3\n", idf_stats)
    print("2\n  2.1\n", ret_ocurrences('barata', idf_stats))
    print("  2.1\n", ret_mul_ocurrences(('a', 'barata', 'diz', 'que'), idf_stats))
    print("3", similarity(('a', 'barata', 'diz', 'que', 'sapato'), idf_stats))



if __name__ == '__main__':
    main()