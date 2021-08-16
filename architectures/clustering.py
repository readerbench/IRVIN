from pathlib import Path
import pandas as pd
import gensim
import numpy as np
import sklearn
from sklearn import decomposition
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import csv
from operator import itemgetter
from tqdm import tqdm
from heapq import nlargest


#This script allows to compute clustering on embedded (by Count or TF-IDF) articles, to confirm duality. 

#Opening of relevant files, and Count or TF-IDF embedding of those articles. 
pca = decomposition.PCA()
store = pd.HDFStore("../storage/table_countvect_clustering/storage_embeddings.h5")
df_train = store["df_counts_train"]
df_test = store["df_counts_test"]
df = pd.concat([df_train, df_test])
filenames = []
with open('../storage/table_countvect_clustering/filenames.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
         filenames.append(row[0].replace(',', ''))
mask = np.where(df['labels']==1)[0]
filenames = itemgetter(*mask)(filenames)
df = df[df.labels==1]
list_col_init = [x for x in df.columns if (np.count_nonzero(df[x]) > 1 and x != "labels")]
X = df[list_col_init]
wv = gensim.models.KeyedVectors.load(f'../storage/FastText_250/FastText_250.wordvectors', mmap = 'r')

word_embeddings = np.array([wv[x] for x in list_col_init])
termdoc = normalize(X.to_numpy(),'l1') 
c = np.einsum('ij,ki->kj', word_embeddings, termdoc)#on multiplie l'emb du mot i avec le count puis on somme 

"""
Don't comment if you want to plot the explained variance with different number of components kept.
pca.n_components = 200
d = pca.fit_transform(c)
percentage_var_exp = pca.explained_variance_/np.sum(pca.explained_variance_)
cum_var_explained = np.cumsum(percentage_var_exp)
print(cum_var_explained)
plt.plot(cum_var_explained)
plt.hlines(0.8, 0, 200, colors = 'r', label = '80 %')
plt.xlabel("n_components")
plt.ylabel("variance ratio explained")
plt.title('variance ratio PCA')
plt.savefig('../results/variance ratio explained')

"""
#On choisit n_components = 35
pca.n_components = 35
reduced_array = pca.fit_transform(c)
"""
Don't comment if you want to see how silhouette scores vary with the number of clusters. 
liste_silh = []
for k in tqdm(range(2, 200)):
    liste = [[None] for _ in range(k)]
    labels = KMeans(n_clusters=k, random_state=0).fit_predict(reduced_array)
    silh = sklearn.metrics.silhouette_score(reduced_array, labels)
    liste_silh.append(silh)
    print(f"k={k}, mean silhouette score : {silh}")
    liste = [np.sum(termdoc[labels == i], axis = 0) for i in range(k)]
idxs = nlargest(15, enumerate(liste_silh), key=lambda x: x[1])
print(idxs)
"""


#km = KMeans(n_clusters=k, random_state=0).fit(reduced_array)
ap = AffinityPropagation().fit(reduced_array)
labels = ap.predict(reduced_array)
k = np.max(labels)
print(k)
#labels = km.predict(reduced_array)
#silh = sklearn.metrics.silhouette_score(reduced_array, labels)
#print(f"k={k}, mean silhouette score : {silh}")
liste = [np.sum(termdoc[labels == i], axis = 0) for i in range(k)]
for i in range(k): 
    #take the most frequents (n_largest); get the corresponding words, and print the 10 most frequent in each cluster.
    n_largest = nlargest(10, range(len(liste[i])), key=lambda x: liste[i][x])
    print(n_largest)
    words = list(itemgetter(*n_largest)(X.columns))
    print(f'cluster : {k}, most frequent : {words}')

closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, reduced_array)
liste_textes = []
for cluster in [0, 1] : 
    liste_textes.append(np.argsort(km.transform(reduced_array)[:,cluster])[:5])
print(f'cluster 0 : {itemgetter(*liste_textes[0])(filenames)}')
print(f'cluster 1 : {itemgetter(*liste_textes[1])(filenames)}')
