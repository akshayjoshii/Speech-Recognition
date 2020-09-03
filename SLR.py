__author__ = "Akshay Joshi"
__email__ = "s8akjosh@stud.uni-saarland.de"

import csv
import itertools
import numpy as np
from collections import defaultdict

# Confusion Matrix, visualization from Matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TSNE, MDS, PCA, DBSCAN, Cosine Sim from Scikit-learn
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics.pairwise import cosine_similarity

# Agglomerative Clustering from Scipy
from scipy.cluster.hierarchy import dendrogram, linkage

class SLR:
    def __init__(self):
        self.file_path = "phoneme_embeddings.tsv"
        self.colours = {} 
        self.colours[0] = 'r'
        self.colours[1] = 'g'
        self.colours[2] = 'b'
        self.colours[-1] = 'k'


    # Generate a confusion matrix/heat map of the cosine similariities of phoneme vector pairs.
    def plotHeatMap(self, final_sim_vectors, final_phoneme_labels):
        fig, ax = plt.subplots()
        im = ax.imshow(final_sim_vectors)
        ax.set_xticks(np.arange(len(final_phoneme_labels)))
        ax.set_yticks(np.arange(len(final_phoneme_labels)))
        ax.set_xticklabels(final_phoneme_labels)
        ax.set_yticklabels(final_phoneme_labels)
        ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Heatmap of Cosine Similarities of Phoneme Vectors", rotation= -90, va = "bottom")
        fig.tight_layout()
        plt.show()
    

    # Return list of lists of phoneme vectors
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
        for index, value in enumerate(pairwise_sim_results, start=1):
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


    # Function to perform Agglomerative/Hierarchial Clustering
    def clusterAnalysis(self, data, labels, linkagecriteria):
        plot_labels = [l for l in labels]
        linkage_method = linkage(data, linkagecriteria)
        plt.figure(figsize = (10, 8))
        plt.xlabel('Phonemes')
        plt.ylabel('Euclidean Distance')
        plt.title(f'{linkagecriteria.upper()} Linkage', fontsize = 25)
        dendrogram(linkage_method, orientation = 'top', labels = plot_labels, distance_sort = 'descending', leaf_rotation = 0.0)
        plt.show()
    

    def executeMultipleLinkages(self, data, labels):
        self.clusterAnalysis(data, labels, 'ward')
        self.clusterAnalysis(data, labels, 'complete')
        self.clusterAnalysis(data, labels, 'average')
        self.clusterAnalysis(data, labels, 'single')


    # Function to plot all the possible PCs and their respective percentage of variance captured
    def multiplePrinicipleComponentsPlot(self, data):
        pca = PCA(n_components = 50)
        pca_data = pca.fit_transform(data)
        # Generating a plot to visualize the number of Priniciple Components
        # required to capture majority (90%) of variance.
        percentage_variance_explained = (pca.explained_variance_) / np.sum(pca.explained_variance_)
        cumulative_variance_explained = np.cumsum(percentage_variance_explained)
        plt.figure(figsize = (8,6))
        plt.title('Percentage of Variance v/s No. of PCs', fontsize = 25)
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
        print(f"The Explained Variance Ratio by 3 Priniciple Components is: {sum(pca.explained_variance_ratio_) * 100} %")
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
        plt.title('Variance Explained v/s No. of Independent Components')
        plt.xlabel('Number of Independent Components')
        plt.ylabel('Variance')
        
        for sig in ica_data.T:
            plt.plot(sig)

        plt.grid()
        plt.show()
    

    # Experimenting with t-SNE (Manifold Learning methods) to perform non-linear dimensionality reduction
    def tStochasticNeighborEmbedding(self, data, labels):
        # When n_components = 2, best value for perplexity is found to be 17 (w/ LR: 300) after performing a grid search.
        # When n_components = 3, best value for perplexity if found to be 37.
        
        #for i in range(5, 51):
            tsne = TSNE(n_components = 2, n_iter = 1000, learning_rate = 300, perplexity = 17, verbose = 1)
            tsne_data = tsne.fit_transform(data)
            tsne_data = np.vstack((tsne_data.T, labels)).T
            fig = plt.figure(figsize = (8,6))
            ax = fig.add_subplot(1,1,1) #, projection = '3d')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            #ax.set_zlabel('Principal Component 3')
            ax.set_title('t-SNE - Two Components', fontsize = 25)
            
            for i, target in enumerate(tsne_data):
                ax.scatter(float(tsne_data[i][0]), float(tsne_data[i][1])) #, float(tsne_data[i][2]))

            ax.grid()
            plt.show()


    # Implementation of MDS (Metric type) to perform non-linear dimensionality reduction
    def multiDimensionalScaling(self, data, labels):
        mds = MDS(n_components = 3, metric = True, n_init = 4, random_state = 0, verbose = 1)
        mds_data = mds.fit_transform(data)
        mds_data = np.vstack((mds_data.T, labels)).T
        fig = plt.figure(figsize = (8,6))
        ax = fig.add_subplot(1,1,1, projection = '3d')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title('MDS - Two Components', fontsize = 25)
        
        for i, target in enumerate(mds_data):
            ax.scatter(float(mds_data[i][0]), float(mds_data[i][1]), float(mds_data[i][2]))

        ax.grid()
        plt.show()
    

    def performDBSCAN(self, data):
        # Reducing the dimension of the data from 50 to 2 using PCA
        pca = PCA(n_components = 2)
        pca_data = pca.fit_transform(data)

        # Loop to perform a grid search to detect best values of 'eps' and 'min_samples' parameter
        """for i in range(5, 50):
            for j in [0.1, 0.2, 0.3, 0.4, 0.5]:"""

        # The best value (to get multiple/unique clusters) for eps is: 0.5 
        # and min_samples is: 9
        dbs = DBSCAN(eps = 0.5, min_samples = 9, metric = 'euclidean')
        dbs_data = dbs.fit(pca_data)

        # print(i, j, dbs_data.labels_)
        labels = dbs_data.labels_
        colour_swatch = [self.colours[label] for label in labels] 
        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principle Component 1')
        ax.set_ylabel('Principle Component 2')
        ax.set_title('DBSCAN Clustering', fontsize = 25) 

        # Plot the DBSCAN clusters
        for i, target in enumerate(pca_data): 
            plt.scatter(pca_data[i][0], pca_data[i][1], color = colour_swatch[i])
        
        # Plotting the legend for Clusters
        r = plt.scatter(pca_data[0], pca_data[1], color = 'r') 
        g = plt.scatter(pca_data[0], pca_data[1], color = 'g') 
        b = plt.scatter(pca_data[0], pca_data[1], color = 'b') 
        k = plt.scatter(pca_data[0], pca_data[1], color = 'k')
        ax.legend((r, g, b, k), ('Cluster 1', 'Cluster 2', 'Cluster 3', 'Couldnt Cluster'))
        ax.grid()
        plt.show()



# Driver Code    
if __name__ == "__main__":
    # Create an object of SLR class
    task = SLR()

    # Generate Phoneme vector & Phoneme pairwise cosine similarity dictionaries
    similarity_dictionary, phoneme_dictionary = task.pairwiseSimilarityDictionary()

    # Parse the phoneme dictionary to retrieve a list of lists of all the phoneme vectors
    phoneme_vectors, phoneme_labels = task.parsePhonemeDictionary(phoneme_dictionary)

    # Parse the Phoneme pair cosine similarity dictionary to retrieve a list of lists of all the pairwise cosine sim vectors
    final_sim_vectors, final_phoneme_labels = task.finalPhonemeSimilaritiesList(similarity_dictionary)

    
    print("-----------------------------------------------------------------------")
    print("The available functions are:")
    print("\n1. Pairwise Cosine Similarity Heatmap/Confusion Matrix")
    print("\n2. Agglomerative Clustering & Dendrogram Visualization")
    print("\n3. Priniciple Component Analysis (PCA)")
    print("\n4. Independent Component Analysis (ICA)")
    print("\n5. t-Distributed Stochastic Neighbor Embedding (t-SNE)")
    print("\n6. Multidimensional Scaling (MDS - Metric)")
    print("\n7. PCA - DBSCAN Clustering")
    print("\n8. Exit")
    print("-----------------------------------------------------------------------")

    while(True):
        user_input = int(input("Please enter the desired function between 1 - 8:\n"))

        if user_input == 1:
            # Plot a Heatmap/Confusion matrix of pairwise cosine similarity values
            task.plotHeatMap(final_sim_vectors, final_phoneme_labels)

        if user_input == 2:
            # Perform Agglomerative clustering and plot corresponding dendrograms
            task.executeMultipleLinkages(phoneme_vectors, phoneme_labels)

        if user_input == 3:
            # Perform 'Priniciple Component Analysis (PCA) for dimentionality reduction and visualization
            task.prinicipleComponentAnalysis(phoneme_vectors, phoneme_labels)

        if user_input == 4:
            # Perform 'Independent Component Analysis (ICA)' for dimentionality reduction and visualization
            task.independentComponentAnalysis(phoneme_vectors)

        if user_input == 5:
            # Perform 't-Distributed Stochastic Neighbor Embedding (t-SNE)' for dimentionality reduction and visualization
            task.tStochasticNeighborEmbedding(phoneme_vectors, phoneme_labels)

        if user_input == 6:
            # Perform 'Multidimensional Scaling (MDS - Metric)' for dimentionality reduction and visualization
            task.multiDimensionalScaling(phoneme_vectors, phoneme_labels)

        if user_input == 7:
            # Perform clustering with DBSCAN
            task.performDBSCAN(phoneme_vectors)
        
        if user_input == 8:
            break
    
