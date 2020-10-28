import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import main

DISCOUNT_FACTOR = [1.2, 3]


metric = "fbeta@10"
recall_view = pd.read_json(f'{main.COLLECTION_PATH}{"BM25tune_results_stemming_stopwords.json"}', orient='split').pivot("k1", "b", metric)
plt.figure(figsize=tuple([s // DISCOUNT_FACTOR[i] for i, s in enumerate(recall_view.shape[::-1])]))
plot = sns.heatmap(recall_view,
            annot=True,
            xticklabels=recall_view.columns.values.round(2),
            yticklabels=recall_view.index.values.round(2)
           )
plot.set_yticklabels(plot.get_yticklabels(), rotation=0)
plt.title(metric, fontsize =20)
plt.show()