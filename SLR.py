__author__ = "Akshay Joshi"
__email__ = "s8akjosh@stud.uni-saarland.de"

import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage

class SLR:
    def __init__(self):
        self.file_path = "phoneme_embeddings.tsv"

    # Generate a confusion matrix/heat map of the cosine similariities of phoneme vector pairs.
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
    # cosine similarities against other phoneme vectors as value.
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

        #print(pairwise_sim_results)
        print(f"Total number of phoneme pairs for cosine similarity: {len(pairwise_sim_results)}")
        similarity_dictionary = defaultdict(list)
        for index, value in enumerate(pairwise_sim_results, start = 1):
            similarity_dictionary[value[0]].append(value[2])
        
        return similarity_dictionary, phoneme_dict
    

    # Return a final reshaped 2D numpy array consisting of pairwise cosine similarities of phoneme embeddings 
    # extracted from cosine similarity dictionary.
    def finalPhonemeSimilaritiesList(self, similarity_dictionary):
        final_phoneme_labels = [key for key, value in similarity_dictionary.items()]

        # Converting the final similarity vector list into numpy array for easier reshaping into a 2D matrix.
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


    def multiplePrinicipleComponentsPlot(self, data):
        pca = PCA(n_components = 50)
        pca_data = pca.fit_transform(data)
        # Generating a plot to visualize the number of Priniciple Components
        # required to capture majority (90%) of variance.
        percentage_variance_explained = (pca.explained_variance_) / np.sum(pca.explained_variance_)
        cumulative_variance_explained = np.cumsum(percentage_variance_explained)
        plt.figure(figsize = (8,6))
        plt.title('Percentage of Variance v/s No. of PCs')
        plt.xlabel('Number of Principle Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.plot(cumulative_variance_explained)
        plt.grid()
        plt.show()


    # Not normalizing the phoneme vectors to preserve certain phoneme significance, consistency & importance.
    def prinicipleComponentAnalysis(self, data, labels):
        # Before proceeding let's visualize the number of Priniciple Components
        # required to capture majority of the variance.
        self.multiplePrinicipleComponentsPlot(data)
        pca = PCA(n_components = 3)
        pca_data = pca.fit_transform(data)
        pca_data = np.vstack((pca_data.T, labels)).T
        print(f"The Explained Variance Ratio by 3 PCs is: {sum(pca.explained_variance_ratio_) * 100} %")
        print(f"Singular Values of 3 PCs are: {pca.singular_values_}")
        fig = plt.figure(figsize = (10,8))
        ax = fig.add_subplot(1,1,1, projection = '3d') 
        ax.set_xlabel('Principle Component 1')
        ax.set_ylabel('Principle Component 2')
        ax.set_zlabel('Principle Component 3')
        ax.set_title('3 Principle Components', fontsize = 25)

        for i, target in enumerate(pca_data):
            ax.scatter(float(pca_data[i][0]), float(pca_data[i][1]), float(pca_data[i][2]))
        ax.grid()
        plt.show()


    # Implementing Fast Independent Componenet Analysis to handle data without significant correlation
    def independentComponentAnalysis(self, data):
        ica = FastICA(n_components = 40, random_state = 0)
        ica_data = ica.fit_transform(data)
        plt.figure(figsize = (8,6))
        plt.title('Variance Explained v/s No. of ICs')
        plt.xlabel('Number of Independent Components')
        plt.ylabel('Variance')
        for sig in ica_data.T:
            plt.plot(sig)
        plt.grid()
        plt.show()
    

    def tStochasticNeighborEmbedding(self, data, labels):
        # When n_components = 2, best value for perplexity is found to be 17 (w/ LR: 300) 
        # after performing a grid search.
        # When n_components = 3, best value for perplexity if found to be 37.
        
        #for i in range(5, 51):
            tsne = TSNE(n_components = 2, n_iter = 1000, learning_rate = 300, perplexity = 17, verbose = 1)
            tsne_data = tsne.fit_transform(data)
            tsne_data = np.vstack((tsne_data.T, labels)).T
            fig = plt.figure(figsize = (8,6))
            ax = fig.add_subplot(1,1,1) #, projection = '3d')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            #ax.set_zlabel('Principal Component 3')
            ax.set_title('Two Components', fontsize = 25)
            for i, target in enumerate(tsne_data):
                ax.scatter(float(tsne_data[i][0]), float(tsne_data[i][1])) #, float(tsne_data[i][2]))
            ax.grid()
            plt.show()


    def multiDimensionalScaling(self, data, labels):
        pass

    
if __name__ == "__main__":
    task = SLR()
    similarity_dictionary, phoneme_dictionary = task.pairwiseSimilarityDictionary()
    phoneme_vectors, phoneme_labels = task.parsePhonemeDictionary(phoneme_dictionary)
    final_sim_vectors, final_phoneme_labels = task.finalPhonemeSimilaritiesList(similarity_dictionary)
    task.plotHeatMap(final_sim_vectors, final_phoneme_labels)
    task.executeMultipleLinkages(phoneme_vectors, phoneme_labels)
    task.prinicipleComponentAnalysis(phoneme_vectors, phoneme_labels)
    task.independentComponentAnalysis(phoneme_vectors)
    task.tStochasticNeighborEmbedding(phoneme_vectors, phoneme_labels)
    
