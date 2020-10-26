import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import main

metric = "mrr@10"

plt.figure(figsize=(10,15))
recall_view = pd.read_json(f'{main.COLLECTION_PATH}{"BM25tune_results.json"}', orient='split').pivot("k1", "b", metric)
plot = sns.heatmap(recall_view,
            annot=True,
            xticklabels=recall_view.columns.values.round(2),
            yticklabels=recall_view.index.values.round(2)
           )
plot.set_yticklabels(plot.get_yticklabels(), rotation=0)
plt.title(metric, fontsize =20)
plt.show()