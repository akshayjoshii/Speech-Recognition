# Programming Challenge in Multilingual Spoken Language Recognition

## Website:
[Dedicated GitHub.io Website](https://akshayjoshii.github.io/Automated_Speech_Recognition/)

## Introduction:

The .tsv file contains phoneme vectors, or phoneme embeddings,  that were obtained from a neural model of grapheme-to-phoneme (g2p) conversion. Each line in the file is a phoneme embedding, where the first entry in each line is the phoneme symbol in IPA,  the rest of the 236 entries in each line are real-value numbers that represent the corresponding 236-dimensional vector. 
 

## Execution Instruction:

1. Install Python 3
2. Install PIP
3. Run "pip install -r requirements.txt"
4. Run "python SLR.py"


## Implemented Functions:

1. Pairwise Cosine Similarity Heatmap/Confusion Matrix
2. Agglomerative Clustering & Dendrogram Visualization
3. Priniciple Component Analysis (PCA)
4. Independent Component Analysis (ICA)
5. t-Distributed Stochastic Neighbor Embedding (t-SNE)
6. Multidimensional Scaling (MDS - Metric)
7. PCA - DBSCAN Clustering


##  Visualizations:

### Cosine Similarity Heatmap:

![Heatmap of pairwise cosine similarity of phoneme vectors](images/heatmap.png?raw=true "Cosine Similarity Heatmap")

### Agglomerative Clustering Dendrograms:

![ward](images/ward.png?raw=true "Ward Linkage")
![complete](images/complete.png?raw=true "Complete Linkage")
![average](images/average.png?raw=true "Average Linkage")
![single](images/single.png?raw=true "Single Linkage")


### Principle Component Analysis:

#### No. of PCs v/s Cumulative Variance:

![Cumulative Variance](images/pca_cumulative_variance.png?raw=true "Cumulative Variance")

#### 50 dimention data points reduced to 3 using 3 Priciple Components:

![PCA](images/pca.png?raw=true "3 Priciple Components")

### Independent Component Analysis:

![ICA](images/ica.png?raw=true "2 Independent Components")

### t-Distributed Stochastic Neighbor Embedding (t-SNE) - Manifold Learning:

![tsne](images/tsne.png?raw=true "2 t-SNE components")

### Multidimensional Scaling (MDS - Metric) - Manifold Learning:

![mds](images/mds.png?raw=true "2 MDS components")

### DBSCAN Clustering:

![dbscan](images/dbscan.png?raw=true "PCA - DBSCAN Clustering")


## Tasks:

1.  Conduct a small research on phoneme embeddings before you start solving the problem.
2.	Reading the .tsv into a suitable data structure (e.g., Pandas data frame, Python dictionary, Numpy array, etc.)
3.	Computing the pair-wise cosine similarity between the phonemes represented by the embeddings and obtaining a confusion matrix of similarity scores. 
4.	Exploring the embeddings space with at least two techniques. We recommend using dimensionality reduction and visualizatio (e.g., PCA, t-SNE), as well as a hierarchical cluster analysis to obtain a dendrogram. 
5.	Writing a 2-3 page report, including the 2 figures, presenting your solution and summarizing your findings. We expect that you would attempt to answer and discuss the question: What do the embeddings really represent? do similarly sounding phonemes have similar embeddings?


## Author
### Akshay Joshi [Universität des Saarlandes]
