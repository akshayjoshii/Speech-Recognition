# Programming Challenge in Automated Speech Recognition

## Introduction:

In the attachment, you will find a ZIP file which consists of a .tsv file. The .tsv file contains phoneme vectors, or phoneme embeddings,  that were obtained from a neural model of grapheme-to-phoneme (g2p) conversion. Each line in the file is a phoneme embedding, where the first entry in each line is the phoneme symbol in IPA,  the rest of the 236 entries in each line are real-value numbers that represent the corresponding 236-dimensional vector. 


## Tasks:

1.  Conduct a small research on phoneme embeddings before you start solving the problem.
2.	Reading the .tsv into a suitable data structure (e.g., Pandas data frame, Python dictionary, Numpy array, etc.)
3.	Computing the pair-wise cosine similarity between the phonemes represented by the embeddings and obtaining a confusion matrix of similarity scores. 
4.	Exploring the embeddings space with at least two techniques. We recommend using dimensionality reduction and visualizatio (e.g., PCA, t-SNE), as well as a hierarchical cluster analysis to obtain a dendrogram. 
5.	Writing a 2-3 page report, including the 2 figures, presenting your solution and summarizing your findings. We expect that you would attempt to answer and discuss the question: What do the embeddings really represent? do similarly sounding phonemes have similar embeddings? 

