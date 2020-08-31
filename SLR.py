import os
import csv
import itertools
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

pfile = open("phoneme_embeddings.tsv", encoding="utf-8")
reader = csv.reader(pfile, delimiter='\t')
phoneme_dict = {}
for row in reader:
    phoneme_dict[row[0]] = row[1:]

pairwise_sim_results = []
counter = 0
for i in phoneme_dict:
    counter += 1
    outer = np.array(phoneme_dict[i]).reshape(1, -1)
    for j in phoneme_dict:
        inner = np.array(phoneme_dict[j]).reshape(1, -1)
        similarity = cosine_similarity(outer, inner)
        res = list(itertools.chain.from_iterable(similarity))
        pairwise_sim_results.append([i, j, res[0]])
    # if counter == 2:
    #     break

pprint(pairwise_sim_results)
print(len(pairwise_sim_results))

similarity_dictionary = defaultdict(list)
for index, value in enumerate(pairwise_sim_results, start = 1):
    similarity_dictionary[value[0]].append(value[2])

final_sim_vectors = [value for key, value in similarity_dictionary.items()]
final_phoneme_labels = [key for key, value in similarity_dictionary.items()]
#print(len(final_phoneme_labels), final_phoneme_labels)

final_sim_vectors = np.asarray(final_sim_vectors).reshape(50, 50)

fig, ax = plt.subplots()
im = ax.imshow(final_sim_vectors)
ax.set_xticks(np.arange(len(final_phoneme_labels)))
ax.set_yticks(np.arange(len(final_phoneme_labels)))
ax.set_xticklabels(final_phoneme_labels)
ax.set_yticklabels(final_phoneme_labels)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Heatmap of Cosine Similarities of Phoneme Vectors", rotation=-90, va="bottom")
fig.tight_layout()
plt.show()