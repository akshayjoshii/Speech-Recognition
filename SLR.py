import os
import csv
import itertools
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage

class SLR:
    def __init__(self):
        self.file_path = "phoneme_embeddings.tsv"

    # Generate a confusion matrix/heat map of the cosine similariities of phoneme vector pairs
    def plotHeatMap(self, final_sim_vectors, final_phoneme_labels):
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
    
    def parsePhonemeDictionary(self, phoneme_dictionary):
        phoneme_labels = [key for key, value in phoneme_dictionary.items()]
        phoneme_vectors = [value for key, value in phoneme_dictionary.items()]

        return phoneme_vectors, phoneme_labels

    # Generate a dictionary with keys as the phonemes and their corresponding row vector containing 
    # cosine similarities against other phoneme vectors as value 
    def pairwiseSimilarityDictionary(self):
        pfile = open(self.file_path, encoding="utf-8")
        reader = csv.reader(pfile, delimiter='\t')
        phoneme_dict = {}
        pairwise_sim_results = []
        for row in reader:
            phoneme_dict[row[0]] = row[1:]

        for i in phoneme_dict:
            outer = np.array(phoneme_dict[i]).reshape(1, -1)
            for j in phoneme_dict:
                inner = np.array(phoneme_dict[j]).reshape(1, -1)
                similarity = cosine_similarity(outer, inner)
                res = list(itertools.chain.from_iterable(similarity))
                pairwise_sim_results.append([i, j, res[0]])

        #pprint(pairwise_sim_results)
        print(f"Total number of phoneme pairs: {len(pairwise_sim_results)}")
        similarity_dictionary = defaultdict(list)
        for index, value in enumerate(pairwise_sim_results, start = 1):
            similarity_dictionary[value[0]].append(value[2])
        
        return similarity_dictionary, phoneme_dict
    

    # Return a final reshaped 2D numpy array consisting of pairwise cosine similarities of phoneme embeddings 
    # extracted from cosine similarity dictionary
    def finalPhonemeSimilaritiesList(self, similarity_dictionary):
        final_phoneme_labels = [key for key, value in similarity_dictionary.items()]

        # Converting the final similarity vector list into numpy array for easier reshaping into a 2D matrix
        final_sim_vectors = [value for key, value in similarity_dictionary.items()]
        final_sim_vectors = np.asarray(final_sim_vectors).reshape(50, 50)

        return final_sim_vectors, final_phoneme_labels


    def clusterAnalysis(self, data, labels, linkagecriteria):
        plot_labels = [l for l in labels]
        linkage_method = linkage(data, linkagecriteria)
        plt.figure(figsize = (10, 8))
        plt.xlabel('Phonemes')
        plt.ylabel('Euclidean Distance')
        plt.title(f'{linkagecriteria.upper()} Linkage')
        dendrogram(linkage_method, orientation = 'top', labels = plot_labels, distance_sort = 'descending') #leaf_rotation = 35.0)
        plt.show()
    

    def executeMultipleLinkages(self, data, labels):
        self.clusterAnalysis(data, labels, 'ward')
        self.clusterAnalysis(data, labels, 'complete')
        self.clusterAnalysis(data, labels, 'average')
        self.clusterAnalysis(data, labels, 'single')


    def prinicipleComponentAnalysis(self, data):
        pca = PCA(n_components=2)
        pca.fit(data)
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)


    
if __name__ == "__main__":
    task = SLR()
    similarity_dictionary, phoneme_dictionary = task.pairwiseSimilarityDictionary()
    phoneme_vectors, phoneme_labels = task.parsePhonemeDictionary(phoneme_dictionary)
    final_sim_vectors, final_phoneme_labels = task.finalPhonemeSimilaritiesList(similarity_dictionary)
    task.plotHeatMap(final_sim_vectors, final_phoneme_labels)
    task.executeMultipleLinkages(phoneme_vectors, phoneme_labels)
    task.prinicipleComponentAnalysis(phoneme_vectors)
    
