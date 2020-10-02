from os import getcwd, listdir
from os.path import isfile, join

from nltk.tokenize import RegexpTokenizer,sent_tokenize
from nltk.tag import SennaTagger
from nltk import ne_chunk, pos_tag, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd



'''--------------------------------------Exercise 1------------------------------------------------------------------'''

''' Exercise sheet Lab1_TextProcessing.pdf '''

#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')

vectorizer = CountVectorizer()

'''----------------SENNA Tagger-------------------'''
sena_tagger = SennaTagger('/home/starksultana/Documentos/MEIC/5o_ano/1o semestre/PRI/Labs/lab1/senna-v3.0/senna')

print("----------------EXERCISE 1-------------------")

#Exercise 1.1

def partition(A,low,high):
    pivot =A[low]
    leftwall = low
    for i in range(low,high+1): #ns s ta bem no pseudo dizia low+1
        if(A[i] < pivot):
            leftwall+=1
            A[leftwall],A[i] = A[i],A[leftwall]
    A[leftwall],A[low] = A[low],A[leftwall]
    return leftwall

def quicksort(A,low,high):
    if len(A) == 1:
        return A
    if (low < high):
        pivotlocation = partition(A,low,high)
        quicksort(A,low,pivotlocation - 1)
        quicksort(A,pivotlocation +1,high)

arr = [10, 7, 8, 4, 1, 5]

n = len(arr)

quicksort(arr,0,n-1)
print("Exercise 1.1: ")
print("Sorted Array: ", arr)

#Exercise 1.2

def read_numbers_file(file):
    with open(file)as f:
        numbers = f.readlines()
    numbers = [nr.strip() for nr in numbers]
    numbers = list(map(int,numbers))
    return numbers

nrs_list = read_numbers_file("ex1.2.txt")
n = len(nrs_list)

quicksort(nrs_list,0,n-1)
print("Exercise 1.2: ")
print("Sorted File Array: ", nrs_list)

#Exercise 1.3

def read_text_file(file):
    dict_words = {}
    with open(file)as f:
        for word in f.read().split():
            if word in dict_words.keys():
                dict_words[word] +=1
            else:
                dict_words[word] = 1
    return dict_words

word_list = read_text_file("ex1.3.txt")
print("Exercise 1.3: ")
print("Word Ocurrence list: ", word_list)

#exercise 1.4
def read_text_file(file):
    words_list = []
    with open(file)as f:
        for word in f.read().split():
            words_list.append(word)
    return words_list

word_list_1 = read_text_file("ex1.3.txt")
word_list_2 = read_text_file("ex1.4.txt")

print("Exercise 1.4: ")
print("Word Intersection list: " , list(set(word_list_1).intersection(word_list_2)))
'''------------------------------------------Exercise 2--------------------------------------------------------------'''
print("----------------EXERCISE 2-------------------")
# Exercise 2.1
def read_text_file_nltk(file):
    with open(file) as txt:
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(txt.read())
    return words

def get_count(word_list):
    dict = {}
    for word in word_list:
        if word in dict.keys():
            dict[word] += 1
        else:
            dict[word] = 1
    return dict

words = read_text_file_nltk("ex1.3.txt")
words_list = get_count(words)

print("Exercise 2.(a): ")
print("Word Ocurrence list: ", words_list)

#Exercise 2.2

def read_text_file_nltk(file):
    with open(file) as txt:
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(txt.read())
    return words

word_list_1 = read_text_file_nltk("ex1.3.txt")
word_list_2 = read_text_file_nltk("ex1.4.txt")

print("Exercise 2.1(b): ")
print("Word Intersection list: " , list(set(word_list_1).intersection(word_list_2)))

#Exercise 2.2

def get_pos_tag(tags_only,tagger):
    only_tags = []
    w_list = read_text_file_nltk("ex1.3.txt")
    if tagger == sena_tagger:
        tags = tagger.tag(w_list)
    else:
        tags = tagger(w_list)
    if tags_only == True:
        for tag in tags:
            only_tags.append(tag[1])
        return only_tags
    return tags

tags = get_pos_tag(tags_only= True,tagger = pos_tag)
tags_occurrences = get_count(tags)

print("Exercise 2.2: ")
print("Syntactic Class Ocurrences: " , tags_occurrences)

#exercise 2.3

print("Exercise 2.3: ")

tags = get_pos_tag(tags_only=False, tagger = pos_tag)
print("Named Entities: ", ne_chunk(tags,binary=True))

#Exercise 2.4/5 #SENNA

print("Exercise 2.4/5: ")
tags = get_pos_tag(tags_only=False, tagger = sena_tagger)
print("Named Entities: ", ne_chunk(tags,binary=True))

#It indeed has some differences, check this later

'''------------------------------------------Exercise 3--------------------------------------------------------------'''
print("----------------EXERCISE 3-------------------")

def scikit_reader(file):
    with open(file,'r') as f:
        text = f.read()
        count = vectorizer.fit_transform([text])
        word_count = pd.DataFrame(count.toarray(), columns = vectorizer.get_feature_names())
        return word_count

print("Exercise 3.1: ")
print("Word counter dataframe: \n", scikit_reader("ex1.3.txt"))

print("Exercise 3.2: ")
print("Sparse data is by nature more easily compressed and thus requires significantly less storage")


def compute_tfidf(file1,file2):
    vectorizer = TfidfVectorizer()
    with open(file1, 'r') as f:
        text1 = f.read()
    with open(file2, 'r') as f:
        text2 = f.read()
    vectors = vectorizer.fit_transform([text1, text2])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    return df

print("Exercise 3.3: ")
print("tfidf dataframe: \n", compute_tfidf("ex1.3.txt","ex1.4.txt"))

def compute_similarity(vectors):
    vector_1 = vectors.iloc[[0]] #1st row
    vector_2 = vectors.iloc[[1]]
    cosineSimilarities = cosine_similarity(vector_1, vector_2).flatten()
    return cosineSimilarities

print("Exercise 3.4: ")
print("Similarity between vectors: \n", compute_similarity(compute_tfidf("ex1.3.txt","ex1.4.txt")))


print("----------------------------------Exercise 4-----------------------------------------------------  ")

#SOLUCAO TIAGO
def parse_word_counts(file_name):
    word_counts = defaultdict(int)

    with open(file_name, encoding="utf8") as f:
        for line in f.readlines():
            for word in line.split():
                word_counts[word] += 1

    return word_counts

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