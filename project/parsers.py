import json
import os
import tarfile
from collections import defaultdict

import numpy as np

from itertools import groupby
from tqdm import tqdm

from bs4 import BeautifulSoup

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
OVERRIDE_SAVED_JSON = False
OVERRIDE_SUBSET_JSON = False
USE_ONLY_EVAL = True
MaxMRRRank = 10
EVAL_SAMPLE_SIZE = 0
BETA = 0.5
K1_TEST_VALS = np.arange(0, 4.1, 0.5)
B_TEST_VALS = np.arange(0, 1.1, 0.2)
DEFAULT_P = 1000
K_TESTS = (1, 3, 5, 10, 20, 50, 100, 200, 500, DEFAULT_P)


def tqdm_generator(members, n):
    for member in tqdm(members, total=n):
        yield member


def read_documents(dirs, sample_size=None):
    docs = {}
    for directory in tqdm(dirs, desc=f'{"PARSING DATASET":20}'):
        for file_name in tqdm(sorted(os.listdir(f"{COLLECTION_PATH}{DATASET}/{directory}"))[:sample_size],
                              desc=f'{f"  DIR[{directory}]":20}', leave=False):
            docs.update(parse_xml_doc(f"{COLLECTION_PATH}{DATASET}/{directory}/{file_name}"))
    return docs


def parse_qrels(filename):
    topic_index, doc_index, topic_index_n, doc_index_n = defaultdict(list), defaultdict(list), defaultdict(
        list), defaultdict(list)
    with open(filename, encoding='utf8') as f:
        for line in tqdm(f.readlines(), desc=f'{"READING QRELS":20}'):
            q_id, doc_id, relevance = line.split()
            if int(relevance.replace('\n', '')):
                topic_index[q_id].append(doc_id)
                doc_index[doc_id].append(q_id)
            else:
                topic_index_n[q_id].append(doc_id)
                doc_index_n[doc_id].append(q_id)

    return dict(topic_index), dict(doc_index), dict(topic_index_n), dict(doc_index_n)


def parse_dataset(split="test"):
    train_dirs, test_dirs = [list(items) for key, items in groupby(sorted(os.listdir(COLLECTION_PATH + DATASET))[:-3],
                                                                   lambda x: x == TRAIN_DATE_SPLIT) if not key]
    train_dirs.append(TRAIN_DATE_SPLIT)
    test_docs = train_docs = None

    if split != 'train':
        test_docs = read_documents(test_dirs[:], sample_size=None)

        print(f"Saving full set to {DATASET}.json...")
        with open(f'{COLLECTION_PATH}{DATASET}_test.json', 'w', encoding='ISO-8859-1') as f:
            f.write(json.dumps(test_docs, indent=4))

        if split == 'test':
            return test_docs

    if split != 'test':
        train_docs = read_documents(train_dirs[:], sample_size=None)

        print(f"Saving full set to {DATASET}.json...")
        with open(f'{COLLECTION_PATH}{DATASET}_train.json', 'w', encoding='ISO-8859-1') as f:
            f.write(json.dumps(train_docs, indent=4))

        if split == 'train':
            return train_docs

    return {'train': train_docs, 'test': test_docs}


def extract_dataset():
    if os.path.isdir(f'{COLLECTION_PATH}{DATASET}'):
        print(f'Directory "{DATASET}" already exists, no extraction needed.', flush=True)
    else:
        print('Directory "rcv1" not found. Extracting "rcv.tar.xz..."', flush=True)
        with tarfile.open('collection/rcv1.tar.xz', 'r:xz') as D:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(D, "collection/", members=tqdm_generator(D,COLLECTION_LEN))


def parse_xml_doc(filename):
    parsed_doc = {}
    with open(filename, encoding='ISO-8859-1') as f:
        soup = BeautifulSoup(f.read().strip(), 'lxml')
        for tag in AVAILABLE_DATA:
            parsed_doc[tag] = ''.join([str(e.string) for e in soup.find_all(tag)])
        return {soup.find(DATA_HEADER[0])[DATA_HEADER[1]]: parsed_doc}


def parse_topics(filename):
    parsed_topics = {}
    with open(filename, encoding='ISO-8859-1') as f:
        soup = BeautifulSoup(f.read().strip(), 'lxml')
        for topic in soup.find_all('top'):
            parsed_topics.update(parse_topic(topic.get_text()))
    return parsed_topics


def parse_topic(topic):
    topic_dict = {}
    contents = topic.split(':')
    id_title = list(filter(None, contents[1].split('\n')))
    topic_dict['title'] = id_title[1].strip()
    topic_dict['desc'] = ''.join(contents[2].split('\n')[:-1]).strip()
    topic_dict['narr'] = ''.join(contents[-1].split('\n')).strip()
    return {id_title[0].strip(): topic_dict}
