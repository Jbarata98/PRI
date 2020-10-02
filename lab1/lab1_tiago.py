from collections import defaultdict
from os import listdir
from os import getcwd
from os.path import isfile, join
import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')

# 1

##1.1

def quicksort(arr, begin, end):
    if end - begin > 1:
        p = partition(arr, begin, end)
        quicksort(arr, begin, p)
        quicksort(arr, p + 1, end)

    return arr


def partition(arr, begin, end):
    pivot = arr[begin]
    i = begin + 1
    j = end - 1

    while True:
        while i <= j and arr[i] <= pivot:
            i = i + 1
        while i <= j and arr[j] >= pivot:
            j = j - 1

        if i <= j:
            arr[i], arr[j] = arr[j], arr[i]
        else:
            arr[begin], arr[j] = arr[j], arr[begin]
            return j


##1.2

def sort_file(file_name):
    val_list = []
    with open(file_name, encoding="utf8") as f:
        for val in f.readlines():
            val_list.append(int(val))

    return quicksort(val_list, 0, len(val_list))


print("1.2:", sort_file("val_list.txt"))


##1.3

def parse_word_counts(file_name):
    word_counts = defaultdict(int)

    with open(file_name, encoding="utf8") as f:
        for line in f.readlines():
            for word in line.replace(",", " ").replace("!", "").split():
                word_counts[word] += 1

    return word_counts


print("1.3:", json.dumps(parse_word_counts("brxt.txt"), indent=4))


# 1.4

def get_common_words_count(file_name_a, file_name_b):
    return len(set(parse_word_counts(file_name_a).keys()).intersection(set(parse_word_counts(file_name_b).keys())))

print("1.4:", get_common_words_count("brxt.txt", "brxt2.txt"))


"""

# 2

##2.1

def nltk_parse_word_counts(file_name):
    word_counts = defaultdict(int)

    with open(file_name, encoding='utf8') as f:
        for sentence in nltk.sent_tokenize(f.read()):
            for word in nltk.word_tokenize(sentence):
                word_counts[word] += 1

    return word_counts


print("1.3:", json.dumps(nltk_parse_word_counts("brxt.txt"), indent=4))


def nltk_get_common_words_count(file_name_a, file_name_b):
    return len(set(nltk_parse_word_counts(file_name_a).keys()).intersection(set(nltk_parse_word_counts(file_name_b).keys())))


print("1.4:", nltk_get_common_words_count("brxt.txt", "brxt2.txt"))

#4 
"""

#3

#3.1

def skl_parse_word_counts(file_name):
    vectorizer = CountVectorizer()
    word_counts = defaultdict(int)

    with open(file_name, encoding="utf8") as f:
        X = vectorizer.fit_transform(f.readlines())

    for n_word, word in enumerate(vectorizer.get_feature_names()):
        for doc in X.toarray():
            word_counts[word] += doc[n_word]

    return word_counts


print("3.1.3:", skl_parse_word_counts("brxt.txt"))


def get_common_words_count(file_name_a, file_name_b):
    return len(set(skl_parse_word_counts(file_name_a).keys()).intersection(set(skl_parse_word_counts(file_name_b).keys())))

print("3.1.4:", get_common_words_count("brxt.txt", "brxt2.txt"))

#3.2 Sparse matrixes differentiates better and can be used as an inverted index


def get_word_index(dir_name):
    inverted_list = defaultdict(list)
    file_path = getcwd() + "/" + dir_name
    for i, file_name in enumerate([f for f in listdir(file_path) if isfile(join(file_path, f))]):
        for word, count in parse_word_counts(dir_name + "/" + file_name).items():
            inverted_list[word].append((i, count))

    return inverted_list

print("4.1:")
for word, indexes in get_word_index("brxts").items():
    print(f'\t{word:10}: {len(indexes):2}: {indexes}')

##4.2

def ii_stats(inverted_index):
    return {"total": max(inverted_index.values(), key=lambda x: max(x, key=lambda x: x[1]))[0][0] + 1,
            "term": len(inverted_index),
            "total_term": len([0 for word in inverted_index.values() if len(word) == 1 and word[0][1] == 1])}

print(ii_stats(get_word_index("brxts")))
