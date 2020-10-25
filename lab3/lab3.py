import os, os.path
from whoosh import index
from whoosh.fields import *
from whoosh.qparser import *

# 1
# 1.1


if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

schema = Schema(id=NUMERIC(stored=True), content=TEXT)
ix = index.create_in("indexdir", schema)
writer = ix.writer()
writer.add_document(id=1, content=u"This is my document!")
writer.add_document(id=2, content=u"This is the second example.")
writer.add_document(id=3, content=u"Examples are many.")
writer.commit()


# 1
# 1.1

def save_index(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir("indexdir")
    schema = Schema(id=NUMERIC(stored=True), content=TEXT(stored=True))  # Schema
    ix = index.create_in("indexdir", schema)
    writer = ix.writer()
    for line in open('pri_cfc.txt', encoding='utf8'):
        doc_id, body_text = line.split(' ', 1)
        writer.add_document(id=int(doc_id), content=body_text)
    writer.commit()

# 1.2

def search_index(string, k=10):
    id_list = []
    ix = index.open_dir("indexdir")
    with ix.searcher() as searcher:
        q = QueryParser("content", ix.schema, group=OrGroup).parse(string)
        results = searcher.search(q, limit=k)
        for r in results:
            id_list.append(r['id'])
    return id_list

# 2
# 2.1

def calc_measures(predicted, expected, metric=None):
    def precision(_predicted, _expected):
        return len(set(_predicted).intersection(set(_expected))) / len(_predicted)

    def recall(_predicted, _expected):
        return len(set(_predicted).intersection(set(_expected))) / len(_expected)

    def f1(_predicted, _expected):
        pre, rec = precision(_predicted, _expected), recall(_predicted, _expected)
        return 0.0 if pre == rec == 0 else 2 * pre * rec / (pre + rec)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    if metric is None:
        metric = metrics.keys()

    return {measure: metrics[measure](predicted, expected) for measure in metric}

#2.2

def read_queries(file):
    with open(file, encoding='utf8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            query, relevant_ids = lines[i:i + 2]
            print(calc_measures(search_index(query), [int(d_id) for d_id in relevant_ids.split()]))

#3
#3.1
#a) 2/5 = 0.4
#b)
def main():
    save_index("indexdir")
    print(search_index(u"John Barate has severe mucus"))
    read_queries('pri_queries.txt')

if __name__ == '__main__':
    main()
