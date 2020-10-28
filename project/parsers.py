import json
import os
import tarfile
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

def parse_dataset():
    train_dirs, test_dirs = [list(items) for key, items in groupby(sorted(os.listdir(COLLECTION_PATH + DATASET))[:-3],
                                                                   lambda x: x == TRAIN_DATE_SPLIT) if not key]
    train_dirs.append(TRAIN_DATE_SPLIT)
    docs = read_documents(test_dirs[:], sample_size=None)

    print(f"Saving full set to {DATASET}.json...")
    with open(f'{COLLECTION_PATH}{DATASET}.json', 'w', encoding='ISO-8859-1') as f:
        f.write(json.dumps(docs, indent=4))
    return docs


def extract_dataset():
    if os.path.isdir(f'{COLLECTION_PATH}{DATASET}'):
        print(f'Directory "{DATASET}" already exists, no extraction needed.', flush=True)
    else:
        print('Directory "rcv1" not found. Extracting "rcv.tar.xz..."', flush=True)
        with tarfile.open('collection/rcv1.tar.xz', 'r:xz') as D:
            D.extractall('collection/', members=tqdm_generator(D, COLLECTION_LEN))


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